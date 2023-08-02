use cimvr_common::{
    glam::Vec2,
    render::{Mesh, MeshHandle, Primitive, Render, UploadMesh, Vertex},
    ui::{egui::DragValue, GuiInputMessage, GuiTab},
    Transform,
};
use cimvr_engine_interface::{make_app_state, pcg::Pcg, pkg_namespace, prelude::*};
use query_accel::QueryAccelerator;
use rand::prelude::*;
use rand_distr::Normal;

mod query_accel;

struct ClientState {
    ui: GuiTab,
    sim: Sim,
    substeps: usize,
}

make_app_state!(ClientState, DummyUserState);

const PARTICLES_RDR: MeshHandle = MeshHandle::new(pkg_namespace!("Cube"));

impl UserState for ClientState {
    fn new(io: &mut EngineIo, sched: &mut EngineSchedule<Self>) -> Self {
        io.create_entity()
            .add_component(Transform::default())
            .add_component(Render::new(PARTICLES_RDR).primitive(Primitive::Points))
            .build();

        let ui = GuiTab::new(io, "MCMC Particle life");

        sched
            .add_system(Self::update_ui)
            .subscribe::<GuiInputMessage>()
            .build();

        sched.add_system(Self::update_sim).build();

        let sim = Sim::new(1000);

        Self { ui, sim, substeps: 100 }
    }
}

impl ClientState {
    fn update_ui(&mut self, io: &mut EngineIo, _query: &mut QueryResult) {
        //cimvr_engine_interface::println!("{}", energy);

        self.ui.show(io, |ui| {
            ui.add(DragValue::new(&mut self.substeps).prefix("Substeps: "));
            ui.add(DragValue::new(&mut self.sim.inverse_temperature).prefix("Temp: ").speed(1e-2));
        });
    }

    fn update_sim(&mut self, io: &mut EngineIo, _query: &mut QueryResult) {
        for _ in 0..self.substeps {
            self.sim.step();
        }

        io.send(&UploadMesh {
            mesh: state_mesh(&self.sim.state),
            id: PARTICLES_RDR,
        });
    }
}

fn state_mesh(state: &State) -> Mesh {
    let vertices = state
        .positions
        .iter()
        .map(|p| Vertex::new([p.x, 0., p.y], [1.; 3]))
        .collect();
    let indices = (0..state.positions.len() as u32).collect();
    Mesh { vertices, indices }
}

struct LennardJones {
    /// Attractive coefficient
    pub attract: f32,
    /// Repulsive coefficient
    pub repulse: f32,
    /// Dispersion energy
    pub dispersion: f32,
}

#[derive(Clone)]
struct State {
    positions: Vec<Vec2>,
}

struct Sim {
    state: State,
    potential: LennardJones,
    accel: QueryAccelerator,
    inverse_temperature: f32,
}

impl Sim {
    pub fn new(n: usize) -> Self {
        let mut rng = rng();

        let s = 0.1;

        let positions = (0..n)
            .map(|_| Vec2::new(rng.gen_range(-s..=s), rng.gen_range(-s..=s)))
            .collect();

        let state = State { positions };

        let potential = LennardJones {
            attract: 0.01,
            repulse: 0.01,
            dispersion: 0.01,
        };

        // We cut off interactions at 5% of the lennard jones potential
        let cutoff = 5./100.;

        let radius = potential.solve(cutoff);
        cimvr_engine_interface::println!("Radius: {}", radius);

        let accel = QueryAccelerator::new(&state.positions, radius);

        let inverse_temperature = 1.;

        Self {
            state,
            potential,
            accel,
            inverse_temperature,
        }
    }

    pub fn step(&mut self) {
        let ref mut rng = rng();

        // Pick a particle
        let idx = rng.gen_range(0..self.state.positions.len());

        // Perterb it
        let original = self.state.positions[idx];
        let mut candidate = original;
        let normal = Normal::new(0.0, 0.001).unwrap();
        candidate.x += normal.sample(rng);
        candidate.y += normal.sample(rng);

        // Calculate the candidate change in energy
        let old_energy = self.energy_due_to(idx, original);
        let new_energy = self.energy_due_to(idx, candidate);
        let delta_e = new_energy - old_energy;

        // Decide whether to accept the change
        let probability = (-self.inverse_temperature * delta_e).exp();
        if probability > rng.gen_range(0.0..=1.0) {
            self.state.positions[idx] = candidate;
            self.accel.replace_point(idx, original, candidate);
        }

    }

    pub fn energy_due_to(&self, idx: usize, pos: Vec2) -> f32 {
        let mut energy = 0.;
        for neighbor in self.accel.query_neighbors(&self.state.positions, idx, pos) {
            let distance = self.state.positions[neighbor].distance(pos);
            let potential = self.potential.eval(distance);
            energy += potential;
        }
        energy
    }
}

fn rng() -> SmallRng {
    let u = ((Pcg::new().gen_u32() as u64) << 32) | Pcg::new().gen_u32() as u64;
    SmallRng::seed_from_u64(u)
}

impl LennardJones {
    /// Returns the potential value given the radius away from the particle
    pub fn eval(&self, radius: f32) -> f32 {
        4. * self.dispersion * ((self.repulse / radius).powi(12) - (self.attract / radius).powi(6))
    }

    /// Solve for the radius given a potential magnitude (sign is discarded)
    /// This is useful for finding an appropriate cutoff radius for local interactions
    /// https://www.desmos.com/calculator/itneqxndwy
    pub fn solve(&self, potential: f32) -> f32 {
        assert!(self.repulse >= 0.);
        assert!(self.attract >= 0.);
        assert!(self.dispersion >= 0.);

        if self.repulse < 0.5 {
            // Corner case where the solution is numerically inaccurate
            self.attract * (4. * self.dispersion / potential).powf(1./6.)
        } else {
            let a = 4. * self.dispersion * self.repulse.powi(12);
            let b = -4. * self.dispersion * self.attract.powi(6);
            let c = -potential;

            // The familiar formula
            let p = (-b - (b.powi(2) - 4. * a * c).sqrt()) / a / 2.;

            p.abs().powf(-1./6.)
        }
    }
}
