#!/usr/bin/env python3
"""Resolve production WGSL assemblies and analyze compute call graphs with Naga."""
from __future__ import annotations
import hashlib, json, re, subprocess, tempfile
from pathlib import Path
from typing import Any

FN_RE=re.compile(r"(?P<attrs>(?:@\w+(?:\([^)]*\))?\s*)*)fn\s+(?P<name>[A-Za-z_]\w*)\s*\(")
CALL_RE=re.compile(r"\b([A-Za-z_]\w*)\s*\(")
RAW_STRING_RE=re.compile(r'r(?P<marks>#{0,8})"(?P<body>.*)"(?P=marks)',re.DOTALL)
STRING_RE=re.compile(r'"(?:\\.|[^"\\])*"',re.DOTALL)
INCLUDE_RE=re.compile(r'^include_str!\(\s*"([^"]+)"\s*\)$',re.DOTALL)
STAGE_RE=re.compile(r"@(compute|fragment|vertex)\b")
STALE_RE=re.compile(r"for\s+now[^\n]{0,80}disabl(?:e|ed)[^\n]{0,40}shadow",re.IGNORECASE)
CONSTRUCTOR_NAMES=("create_labeled_shader_module","create_shader_module")
_RUST_FUNCTIONS:dict[str,list[tuple[Path,str,str]]]={}

def _tracked(root:Path,patterns:tuple[str,...])->list[Path]:
    try: result=subprocess.run(["git","-C",str(root),"ls-files","--",*patterns],check=True,capture_output=True,text=True)
    except (OSError,subprocess.CalledProcessError) as exc: raise ValueError(f"cannot enumerate tracked shader sources: {exc}") from exc
    return [root/line for line in result.stdout.splitlines() if line]

def _balanced(text:str,opening:int,left:str="(",right:str=")")->tuple[str,int]:
    depth=0; quote=None; raw_marks=None; index=opening
    while index<len(text):
        if raw_marks is not None:
            terminator='"'+raw_marks
            if text.startswith(terminator,index): index+=len(terminator); raw_marks=None; continue
            index+=1; continue
        char=text[index]
        if quote:
            if char=="\\": index+=2; continue
            if char==quote: quote=None
            index+=1; continue
        if text.startswith("//",index):
            newline=text.find("\n",index+2); index=len(text) if newline<0 else newline+1; continue
        if text.startswith("/*",index):
            closing=text.find("*/",index+2)
            if closing<0: raise ValueError("unterminated block comment")
            index=closing+2; continue
        raw=re.match(r'r(#{0,8})"',text[index:])
        if raw: raw_marks=raw.group(1); index+=raw.end(); continue
        if char == "\"": quote=char; index+=1; continue
        if char==left: depth+=1
        elif char==right:
            depth-=1
            if depth==0: return text[opening+1:index],index+1
        index+=1
    raise ValueError("unterminated Rust expression")

def _split_args(value:str)->list[str]:
    parts=[]; start=0; depths={"(":0,"[":0,"{":0}; pairs={")":"(","]":"[","}":"{"}; quote=None; raw_marks=None; index=0
    while index<len(value):
        if raw_marks is not None:
            term='"'+raw_marks
            if value.startswith(term,index): index+=len(term); raw_marks=None; continue
            index+=1; continue
        char=value[index]
        if quote:
            if char=="\\": index+=2; continue
            if char==quote: quote=None
            index+=1; continue
        raw=re.match(r'r(#{0,8})"',value[index:])
        if raw: raw_marks=raw.group(1); index+=raw.end(); continue
        if char == "\"": quote=char
        elif char in depths: depths[char]+=1
        elif char in pairs: depths[pairs[char]]-=1
        elif char=="," and not any(depths.values()): parts.append(value[start:index].strip()); start=index+1
        index+=1
    tail=value[start:].strip()
    if tail: parts.append(tail)
    return parts

def _rust_string(value:str)->str|None:
    value=value.strip(); raw=RAW_STRING_RE.fullmatch(value)
    if raw: return raw.group("body")
    if STRING_RE.fullmatch(value):
        try: return json.loads(value)
        except json.JSONDecodeError: return None
    return None

def _assignment(text:str,name:str,before:int)->str|None:
    matches=list(re.finditer(rf"\b(?:let|const|static)\s+(?:mut\s+)?{re.escape(name)}(?:\s*:[^=;]+)?\s*=",text[:before]))
    if not matches: return None
    start=matches[-1].end(); depth=0; quote=None; index=start
    while index<before:
        char=text[index]
        if quote:
            if char=="\\": index+=2; continue
            if char==quote: quote=None
        elif char == "\"": quote=char
        elif char in "([{": depth+=1
        elif char in ")]}": depth-=1
        elif char==";" and depth==0: return text[start:index].strip()
        index+=1
    return None

def _cfg_test_ranges(text:str)->list[tuple[int,int]]:
    ranges=[]
    for match in re.finditer(r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]\s*(?:mod\s+[A-Za-z_]\w*\s*)?\{",text):
        opening=text.find("{",match.start())
        try: _,end=_balanced(text,opening,"{","}")
        except ValueError: continue
        ranges.append((match.start(),end))
    return ranges

def _function_spans(text:str)->list[dict[str,Any]]:
    spans=[]
    for match in re.finditer(r"\bfn\s+([A-Za-z_]\w*)\s*\(",text):
        opening=text.find("(",match.start())
        try:
            parameters,after_parameters=_balanced(text,opening)
            body_start=text.find("{",after_parameters)
            if body_start<0: continue
            _,body_end=_balanced(text,body_start,"{","}")
        except ValueError: continue
        spans.append({"name":match.group(1),"start":match.start(),"parameters":parameters,"body_start":body_start,"body_end":body_end})
    return spans

def _source_parameter(function:dict[str,Any],name:str)->int|None:
    for index,parameter in enumerate(_split_args(function["parameters"])):
        if re.search(rf"\b{re.escape(name)}\s*:",parameter): return index
    return None

def _resolve(expr:str,path:Path,text:str,before:int,constants:dict[str,tuple[Path,str,str]],stack:set[str])->str|None:
    expr=expr.strip().rstrip(",")
    while expr.startswith("&"): expr=expr[1:].strip()
    for suffix in (".as_str()",".as_ref()",".to_string()",".into()"):
        if expr.endswith(suffix): return _resolve(expr[:-len(suffix)],path,text,before,constants,stack)
    if expr.startswith("("):
        try:
            body,end=_balanced(expr,0)
            if end==len(expr): return _resolve(body,path,text,before,constants,stack)
        except ValueError: pass
    literal=_rust_string(expr)
    if literal is not None: return literal
    include=INCLUDE_RE.fullmatch(expr)
    if include:
        target=(path.parent/include.group(1)).resolve()
        if not target.is_file() and "/shaders/" in include.group(1): target=next(parent for parent in path.parents if (parent/"src").is_dir())/"src/shaders"/include.group(1).split("/shaders/",1)[1]
        try: return target.read_text(encoding="utf-8")
        except OSError: return None
    if expr.startswith("concat!("):
        try: body,end=_balanced(expr,expr.index("("))
        except ValueError: return None
        if end!=len(expr): return None
        values=[_resolve(part,path,text,before,constants,stack) for part in _split_args(body)]
        return None if any(value is None for value in values) else "".join(values)  # type: ignore[arg-type]
    if expr.startswith("format!("):
        try: body,end=_balanced(expr,expr.index("("))
        except ValueError: return None
        args=_split_args(body)
        template=_resolve(args[0],path,text,before,constants,stack) if args else None
        values=[_resolve(arg,path,text,before,constants,stack) for arg in args[1:]]
        if template is None or any(value is None for value in values): return None
        for value in values: template=template.replace("{}",value,1)  # type: ignore[arg-type]
        for name in re.findall(r"\{([A-Za-z_]\w*)\}",template):
            value=_resolve(name,path,text,before,constants,stack)
            if value is None: return None
            template=template.replace("{"+name+"}",value)
        return template
    for wrapper in ("wgpu::ShaderSource::Wgsl","ShaderSource::Wgsl","std::borrow::Cow::Borrowed","Cow::Borrowed"):
        if expr.startswith(wrapper+"("):
            try: body,end=_balanced(expr,len(wrapper))
            except ValueError: return None
            return _resolve(body,path,text,before,constants,stack) if end==len(expr) else None
    replace=re.fullmatch(r"(.+)\.replace\((.*)\)",expr,re.DOTALL)
    if replace:
        base=_resolve(replace.group(1),path,text,before,constants,stack); args=_split_args(replace.group(2))
        old=_resolve(args[0],path,text,before,constants,stack) if len(args)==2 else None; new=_resolve(args[1],path,text,before,constants,stack) if len(args)==2 else None
        return base.replace(old,new) if base is not None and old is not None and new is not None else None
    call=re.fullmatch(r"(?:[A-Za-z_]\w*::)*([A-Za-z_]\w*)\s*\((.*)\)",expr,re.DOTALL)
    if call and call.group(1) not in stack:
        candidates=_RUST_FUNCTIONS.get(call.group(1),[])
        for function_path,function_text,body in candidates:
            pieces=[]
            for included in re.finditer(r'(strip_includes\s*\()?include_str!\(\s*"([^"]+\.wgsl)"\s*\)',body):
                target=(function_path.parent/included.group(2)).resolve()
                if not target.is_file() and "/shaders/" in included.group(2): target=next(parent for parent in function_path.parents if (parent/"src").is_dir())/"src/shaders"/included.group(2).split("/shaders/",1)[1]
                try: piece=target.read_text(encoding="utf-8")
                except OSError: pieces=[]; break
                if included.group(1): piece="\n".join(line for line in piece.splitlines() if not line.lstrip().startswith("#include"))
                pieces.append(piece)
            if "CsmRenderer::shader_source()" in body:
                shadow=_resolve("CSM_SHADER_SOURCE",function_path,function_text,len(function_text),constants,stack|{call.group(1)})
                if shadow is None: return None
                replacement=re.search(r'CsmRenderer::shader_source\(\)\.replace\(\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\)',body,re.DOTALL)
                if replacement: shadow=shadow.replace(replacement.group(1),replacement.group(2))
                pieces.insert(0,shadow)
            if "terrain_base(" in body:
                base=_resolve("terrain()",function_path,function_text,len(function_text),constants,stack|{call.group(1)})
                if base is None: return None
                pieces.insert(0,base)
            if pieces: return "\n".join(pieces)
            nested=list(re.finditer(r"(?:[A-Za-z_]\w*::)*([A-Za-z_]\w*)\s*\(([^;{}]*)\)",body,re.DOTALL))
            for nested_call in reversed(nested):
                if nested_call.group(1) in _RUST_FUNCTIONS and nested_call.group(1) not in stack:
                    value=_resolve(nested_call.group(0),function_path,function_text,len(function_text),constants,stack|{call.group(1)})
                    if value is not None: return value
    identifier=expr.split("::")[-1]
    if re.fullmatch(r"[A-Za-z_]\w*",identifier):
        key=f"{path}:{identifier}"
        if key in stack: return None
        local=_assignment(text,identifier,before)
        if local is not None: return _resolve(local,path,text,before,constants,stack|{key})
        record=constants.get(identifier)
        if record:
            other_path,other_text,other_expr=record; return _resolve(other_expr,other_path,other_text,len(other_text),constants,stack|{key})
    return None

def _functions(source:str,label:str)->dict[str,dict[str,Any]]:
    result={}
    for match in FN_RE.finditer(source):
        start=source.find("{",match.end())
        if start<0: continue
        try: body,_=_balanced(source,start,"{","}")
        except ValueError: raise ValueError(f"{label}: unterminated WGSL function {match['name']}")
        result[match["name"]]={"compute":"@compute" in match["attrs"],"calls":set(CALL_RE.findall(body)),"compare":len(re.findall(r"\btextureSampleCompare\s*\(",body))}
    return result

def _locally_assembled_variants(path:Path,text:str,assignment:str,constants:dict[str,tuple[Path,str,str]])->list[str]:
    """Evaluate the tracked DD variant assembler used by locally parsed sites."""
    if not re.match(r"(?:assembled_source|assemble_with_entry)\s*\(",assignment): return []
    candidates=_RUST_FUNCTIONS.get("assemble_with_entry",[])
    if not candidates: return []
    function_path,function_text,body=candidates[0]
    determinism=_resolve("DETERMINISM",function_path,function_text,len(function_text),constants,set())
    shader=_resolve("DD_SHADER",function_path,function_text,len(function_text),constants,set())
    selected=re.findall(r"TwoProdVariant::[A-Za-z_]\w*\s*=>\s*\"([^\"]+)\"",body)
    barrier_match=re.search(r'\.replace\(\s*"__DD_BARRIER_BODY__"\s*,\s*("(?:\\.|[^"\\])*")\s*,?\s*\)',body,re.DOTALL)
    barrier=_rust_string(barrier_match.group(1)) if barrier_match else None
    call_args=_split_args(_balanced(assignment,assignment.index("("))[0])
    entry_name=call_args[1].strip() if assignment.startswith("assemble_with_entry") and len(call_args)>1 else "HARNESS"
    entry=_resolve(entry_name,path,text,len(text),constants,set())
    if determinism is None or shader is None or barrier is None or entry is None or not selected: return []
    return [f"{determinism}\n{shader.replace('__DD_TWO_PROD_CALL__',choice).replace('__DD_BARRIER_BODY__',barrier)}\n{entry}" for choice in selected]

def _naga_analyze(assemblies:dict[str,str])->tuple[int,list[str]]:
    if not assemblies: return 0,[]
    with tempfile.TemporaryDirectory(prefix="nephele-naga-") as directory:
        root=Path(directory); src=root/"src"; src.mkdir()
        (root/"Cargo.toml").write_text('[package]\nname="nephele_naga_check"\nversion="0.0.0"\nedition="2021"\n[dependencies]\nnaga={version="=0.19.2",features=["wgsl-in"]}\n',encoding="utf-8")
        (src/"main.rs").write_text(r'''use naga::{Expression,ShaderStage,Statement}; use std::{collections::HashSet,env,fs};
fn calls(block:&naga::Block,out:&mut Vec<naga::Handle<naga::Function>>){for statement in block{match statement{Statement::Block(v)=>calls(v,out),Statement::If{accept,reject,..}=>{calls(accept,out);calls(reject,out)},Statement::Switch{cases,..}=>for case in cases{calls(&case.body,out)},Statement::Loop{body,continuing,..}=>{calls(body,out);calls(continuing,out)},Statement::Call{function,..}=>out.push(*function),_=>{}}}}
fn main(){for file in env::args().skip(1){let source=fs::read_to_string(&file).unwrap();let module=naga::front::wgsl::parse_str(&source).unwrap_or_else(|e|panic!("{}: {}",file,e.emit_to_string(&source)));naga::valid::Validator::new(naga::valid::ValidationFlags::all(),naga::valid::Capabilities::all()).validate(&module).unwrap_or_else(|e|panic!("{}: {:?}",file,e));for entry in &module.entry_points{if entry.stage!=ShaderStage::Compute{continue}if entry.function.expressions.iter().any(|(_,e)|matches!(e,Expression::ImageSample{depth_ref:Some(_),..})){panic!("{}: compute {} compare",file,entry.name)}let mut pending=Vec::new();calls(&entry.function.body,&mut pending);let mut seen=HashSet::new();while let Some(handle)=pending.pop(){if !seen.insert(handle){continue}let function=&module.functions[handle];if function.expressions.iter().any(|(_,e)|matches!(e,Expression::ImageSample{depth_ref:Some(_),..})){panic!("{}: compute {} compare",file,entry.name)}calls(&function.body,&mut pending)}}}}''',encoding="utf-8")
        ordered=sorted(assemblies.items()); paths=[]
        for index,(_,source) in enumerate(ordered):
            target=root/f"assembly-{index}.wgsl"; target.write_text(source,encoding="utf-8"); paths.append(str(target))
        completed=subprocess.run(["cargo","run","--offline","--quiet","--manifest-path",str(root/"Cargo.toml"),"--",*paths],cwd=root,capture_output=True,text=True)
        if completed.returncode:
            output=(completed.stderr or completed.stdout)[-4000:]; match=re.search(r"assembly-(\d+)\.wgsl",output); label=ordered[int(match.group(1))][0] if match and int(match.group(1))<len(ordered) else "unknown"
            raise ValueError(f"Naga assembly parse/call-graph failure for {label}: {output}")
    return len(paths),sorted(assemblies)

def analyze_shaders(repo_root:Path)->dict[str,Any]:
    repo_root=repo_root.resolve(); wgsl_paths=sorted(_tracked(repo_root,("src/*.wgsl","src/**/*.wgsl"))); rust_paths=sorted(_tracked(repo_root,("src/*.rs","src/**/*.rs")))
    if not wgsl_paths: raise ValueError("no tracked production WGSL shaders found")
    all_wgsl={path.resolve() for path in wgsl_paths}; all_tracked_wgsl={path.resolve() for path in _tracked(repo_root,("*.wgsl","**/*.wgsl"))}; constants={}; rust={}
    for path in rust_paths:
        text=path.read_text(encoding="utf-8"); rust[path]=text
        for match in re.finditer(r"\b(?:const|static)\s+([A-Za-z_]\w*)(?:\s*:[^=;]+)?\s*=",text):
            expression=_assignment(text,match.group(1),len(text))
            if expression is not None: constants[match.group(1)]=(path,text,expression)
    _RUST_FUNCTIONS.clear()
    for path,text in rust.items():
        for match in re.finditer(r"\bfn\s+([A-Za-z_]\w*)\s*\(",text):
            opening=text.find("(",match.start())
            try: _,after_parameters=_balanced(text,opening); opening_body=text.find("{",after_parameters); body,_=_balanced(text,opening_body,"{","}")
            except ValueError: continue
            _RUST_FUNCTIONS.setdefault(match.group(1),[]).append((path,text,body))
    include_edges=[]; embedded_sources={}; shader_source_files=[]; naga_files=[]; construction_files=set(); wrapper_specs={}; wrapper_calls=set(); wrapper_delegations=[]; self_naga=[]; sites=[]; assemblies={}; unresolved=[]
    for path,text in rust.items():
        relative=path.relative_to(repo_root).as_posix()
        test_ranges=_cfg_test_ranges(text)
        if "ShaderSource::Wgsl" in text: shader_source_files.append(relative)
        if "naga::front::wgsl::parse_str" in text: naga_files.append(relative)
        for match in re.finditer(r'include_str!\(\s*"([^"]+\.wgsl)"\s*\)',text):
            target=(path.parent/match.group(1)).resolve()
            if target not in all_wgsl and "/shaders/" in match.group(1): target=(repo_root/"src/shaders"/match.group(1).split("/shaders/",1)[1]).resolve()
            if target in all_tracked_wgsl and target not in all_wgsl: continue
            if target not in all_wgsl: raise ValueError(f"{relative}: WGSL include is not tracked: {match.group(1)}")
            include_edges.append({"rust":relative,"wgsl":target.relative_to(repo_root).as_posix()})
        for index,match in enumerate(re.finditer(r'r(?P<marks>#{0,8})"(?P<body>.*?)"(?P=marks)',text,re.DOTALL),1):
            if STAGE_RE.search(match.group("body")): embedded_sources[f"{relative}::embedded-wgsl-{index}"]=match.group("body")
        for name in CONSTRUCTOR_NAMES:
            if path.stem.endswith("tests"): continue
            for match in re.finditer(rf"\b{re.escape(name)}\s*\(",text):
                if any(start<=match.start()<end for start,end in test_ranges): continue
                if re.search(r"\bfn\s+$",text[max(0,match.start()-12):match.start()]): continue
                try: body,_=_balanced(text,text.find("(",match.start()))
                except ValueError: unresolved.append(f"{relative}:{text.count(chr(10),0,match.start())+1}:unterminated"); continue
                args=_split_args(body); expression=None
                if name=="create_labeled_shader_module" and len(args)>=3: expression=args[2]
                elif name=="create_shader_module":
                    found=re.search(r"\bsource\s*:\s*(?:wgpu::)?ShaderSource::Wgsl\s*\(",body)
                    if found:
                        inner_start=body.find("(",found.start()); expression,_=_balanced(body,inner_start)
                if expression is None: continue
                construction_files.add(relative); line=text.count("\n",0,match.start())+1; label=f"{relative}:{line}:{name}"; sites.append(label)
                bare=expression.strip().lstrip("&").removesuffix(".into()")
                functions=[function for function in _function_spans(text) if function["body_start"]<match.start()<function["body_end"]]
                enclosing=min(functions,key=lambda function:function["body_end"]-function["body_start"]) if functions else None
                parameter_index=_source_parameter(enclosing,bare) if enclosing and re.fullmatch(r"[A-Za-z_]\w*",bare) else None
                if relative=="src/core/shader_registry.rs" or parameter_index is not None:
                    if relative=="src/core/shader_registry.rs":
                        enclosing=min(functions,key=lambda function:function["body_end"]-function["body_start"]) if functions else None
                        parameter_index=_source_parameter(enclosing,"source") if enclosing else None
                    if enclosing is None or parameter_index is None: unresolved.append(f"{label}:wrapper source parameter is not identifiable"); continue
                    parents=[function for function in functions if function is not enclosing and function["body_start"]<enclosing["start"]<function["body_end"]]
                    parent=min(parents,key=lambda function:function["body_end"]-function["body_start"]) if parents else None
                    wrapper_specs[label]={"path":path,"relative":relative,"function":enclosing["name"],"source_parameter_index":parameter_index,"scope":(parent["body_start"],parent["body_end"]) if parent else None}
                    continue
                resolved=_resolve(expression,path,text,match.start(),constants,set())
                assignment=_assignment(text,bare,match.start()) if re.fullmatch(r"[A-Za-z_]\w*",bare) else None
                if resolved is None and assignment and "include_str!" in assignment:
                    variants=[]
                    for included in re.finditer(r'include_str!\(\s*"([^"]+\.wgsl)"\s*\)',assignment):
                        variant=_resolve(included.group(0),path,text,match.start(),constants,set())
                        if variant is not None: variants.append(variant)
                    if variants:
                        for index,variant in enumerate(variants): assemblies[f"{label}#variant-{index}"]=variant
                        continue
                if resolved is None and re.fullmatch(r"[A-Za-z_]\w*",bare):
                    start=max(text.rfind("fn ",0,match.start()),0); preceding=text[start:match.start()]
                    if re.search(rf"naga::front::wgsl::parse_str\s*\(\s*&?{re.escape(bare)}\s*\)",preceding):
                        variants=_locally_assembled_variants(path,text,assignment or "",constants)
                        if variants:
                            for index,variant in enumerate(variants): assemblies[f"{label}#locally-parsed-variant-{index}"]=variant
                        else: self_naga.append(label)
                        continue
                if resolved is None: unresolved.append(f"{label}:{' '.join(expression.split())[:160]}")
                else: assemblies[label]=resolved
    wrapper_sites=set(wrapper_specs)
    for wrapper_label,spec in wrapper_specs.items():
        candidate_paths=rust_paths if spec["function"]=="create_labeled_shader_module" else [spec["path"]]
        for caller_path in candidate_paths:
            caller_text=rust[caller_path]; caller_relative=caller_path.relative_to(repo_root).as_posix()
            if caller_path.stem.endswith("tests"): continue
            caller_test_ranges=_cfg_test_ranges(caller_text)
            for call in re.finditer(rf"\b{re.escape(spec['function'])}\s*\(",caller_text):
                if any(start<=call.start()<end for start,end in caller_test_ranges): continue
                if re.search(r"\bfn\s+$",caller_text[max(0,call.start()-12):call.start()]): continue
                if caller_path==spec["path"] and spec["scope"] and not spec["scope"][0]<call.start()<spec["scope"][1]: continue
                try: body,_=_balanced(caller_text,caller_text.find("(",call.start()))
                except ValueError: unresolved.append(f"{caller_relative}:{caller_text.count(chr(10),0,call.start())+1}:{spec['function']}:unterminated wrapper call"); continue
                arguments=_split_args(body); index=spec["source_parameter_index"]
                call_label=f"{caller_relative}:{caller_text.count(chr(10),0,call.start())+1}:{spec['function']}"
                if index>=len(arguments): unresolved.append(f"{call_label}:missing wrapper source argument"); continue
                expression=arguments[index]; resolved=_resolve(expression,caller_path,caller_text,call.start(),constants,set())
                if resolved is not None:
                    assembly_label=f"{wrapper_label}<-{call_label}"; assemblies[assembly_label]=resolved; wrapper_calls.add(f"{wrapper_label}<-{call_label}"); continue
                bare=expression.strip().lstrip("&").removesuffix(".into()"); assignment=_assignment(caller_text,bare,call.start()) if re.fullmatch(r"[A-Za-z_]\w*",bare) else None
                variants=[]
                if assignment and "include_str!" in assignment:
                    for included in re.finditer(r'include_str!\(\s*"([^"]+\.wgsl)"\s*\)',assignment):
                        variant=_resolve(included.group(0),caller_path,caller_text,call.start(),constants,set())
                        if variant is not None: variants.append(variant)
                if variants:
                    for variant_index,variant in enumerate(variants):
                        assembly_label=f"{wrapper_label}<-{call_label}#variant-{variant_index}"; assemblies[assembly_label]=variant; wrapper_calls.add(f"{wrapper_label}<-{call_label}")
                    continue
                if assignment and re.fullmatch(r"[A-Za-z_]\w*",bare):
                    local_variants=_locally_assembled_variants(caller_path,caller_text,assignment,constants)
                    if local_variants:
                        for variant_index,variant in enumerate(local_variants):
                            assembly_label=f"{wrapper_label}<-{call_label}#locally-parsed-variant-{variant_index}"; assemblies[assembly_label]=variant; wrapper_calls.add(f"{wrapper_label}<-{call_label}")
                        continue
                if call_label in wrapper_sites:
                    wrapper_delegations.append((wrapper_label,call_label)); continue
                unresolved.append(f"{call_label}:unresolved wrapper source argument {' '.join(expression.split())[:160]}")
    for wrapper_label,delegated_wrapper in wrapper_delegations:
        expanded=[(label,source) for label,source in assemblies.items() if label.startswith(delegated_wrapper+"<-")]
        if not expanded:
            unresolved.append(f"{wrapper_label}<-{delegated_wrapper}:wrapper delegation has no resolved live caller")
            continue
        wrapper_calls.add(f"{wrapper_label}<-{delegated_wrapper}")
        for label,source in expanded: assemblies[f"{wrapper_label}<-{label}"]=source
    if unresolved: raise ValueError("unresolved production shader source expressions: "+"; ".join(sorted(unresolved)))
    naga_count,naga_labels=_naga_analyze(assemblies) if (repo_root/"Cargo.toml").is_file() else (0,[])
    sources={path.relative_to(repo_root).as_posix():path.read_text(encoding="utf-8") for path in wgsl_paths}; sources.update(assemblies); sources.update(embedded_sources); violations=[]; stale=0
    for label,source in sources.items():
        stale+=len(STALE_RE.findall(source)); records=_functions(source,label)
        for name,record in records.items():
            if not record["compute"]: continue
            pending=[name]; seen=set()
            while pending:
                current=pending.pop()
                if current in seen or current not in records: continue
                seen.add(current); entry=records[current]
                if entry["compare"]: violations.append(f"{label}::{name}"); break
                pending.extend(entry["calls"])
    return {"schema":"forge3d.nephele.gate5_static/5","analysis_engine":"exact-wrapper-expanded-rust-source-resolution-plus-naga-ir-call-graph-v4","tracked_wgsl_files":[path.relative_to(repo_root).as_posix() for path in wgsl_paths],"rust_wgsl_include_edges":sorted(include_edges,key=lambda value:(value["rust"],value["wgsl"])),"rust_shader_source_files":sorted(set(shader_source_files)),"rust_dynamic_registry_files":sorted({spec["relative"] for spec in wrapper_specs.values()}),"rust_shader_construction_files":sorted(construction_files),"rust_shader_construction_wrappers":sorted(wrapper_specs),"rust_shader_wrapper_invocations":sorted(wrapper_calls),"rust_self_naga_validated_constructions":sorted(self_naga),"rust_shader_construction_expressions":sorted(sites),"resolved_source_sha256":{label:hashlib.sha256(source.encode()).hexdigest() for label,source in sorted(assemblies.items())},"unresolved_source_expressions":sorted(unresolved),"embedded_rust_wgsl_sources":sorted(embedded_sources),"naga_parser_contract_files":sorted(set(naga_files)),"executed_assembled_naga_contracts":naga_labels,"naga_validated_assemblies":naga_count,"compute_entry_texture_sample_compare_calls":len(set(violations)),"violating_compute_entries":sorted(set(violations)),"stale_disabled_shadow_comments":stale}
