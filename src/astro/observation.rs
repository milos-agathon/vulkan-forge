use super::{
    body_position, catalog, moon_phase, time::UtcDateTime, AstroError, Body, BodyPosition,
    MoonPhase, Observer,
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, OnceLock, RwLock,
};

const BODIES: [Body; 7] = [
    Body::Sun,
    Body::Moon,
    Body::Mercury,
    Body::Venus,
    Body::Mars,
    Body::Jupiter,
    Body::Saturn,
];

#[derive(Debug)]
pub struct ObservationSnapshot {
    pub revision: u64,
    pub utc: UtcDateTime,
    pub observer: Observer,
    pub bodies: [(Body, BodyPosition); 7],
    pub moon_phase: MoonPhase,
    pub stars: Vec<catalog::StarInstance>,
}

pub fn set_observation(
    utc: UtcDateTime,
    observer: Observer,
) -> Result<Arc<ObservationSnapshot>, AstroError> {
    static REVISION: AtomicU64 = AtomicU64::new(0);
    let bodies: [(Body, BodyPosition); 7] = BODIES
        .map(|body| body_position(body, utc, observer, false).map(|position| (body, position)))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| AstroError::InvalidData("observation bodies"))?;
    let sun_altitude = bodies[0].1.altitude;
    let snapshot = Arc::new(ObservationSnapshot {
        revision: REVISION.fetch_add(1, Ordering::Relaxed) + 1,
        utc,
        observer,
        bodies,
        moon_phase: moon_phase(utc)?,
        stars: catalog::star_instances(utc, observer, sun_altitude)?,
    });
    *slot().write().expect("observation state poisoned") = Some(snapshot.clone());
    Ok(snapshot)
}

pub fn current_observation() -> Option<Arc<ObservationSnapshot>> {
    slot().read().expect("observation state poisoned").clone()
}

fn slot() -> &'static RwLock<Option<Arc<ObservationSnapshot>>> {
    static OBSERVATION: OnceLock<RwLock<Option<Arc<ObservationSnapshot>>>> = OnceLock::new();
    OBSERVATION.get_or_init(|| RwLock::new(None))
}
