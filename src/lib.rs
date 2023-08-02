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
    potential_cutoff: f32,
    editor_potential: LennardJones,
    n: usize,
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

        let potential_cutoff = 0.001;
        let editor_potential = LennardJones::default();

        let n = 500;

        let sim = Sim::new(n, potential_cutoff, Ruleset::default(), 10.0, 0.0003);

        Self {
            ui,
            sim,
            substeps: 500,
            potential_cutoff,
            editor_potential,
            n,
        }
    }
}

impl ClientState {
    fn update_ui(&mut self, io: &mut EngineIo, _query: &mut QueryResult) {
        //cimvr_engine_interface::println!("{}", energy);

        self.ui.show(io, |ui| {
            ui.add(DragValue::new(&mut self.substeps).prefix("Substeps: "));
            ui.add(
                DragValue::new(&mut self.sim.temperature)
                    .prefix("Temp: ")
                    .speed(1e-2),
            );
            ui.add(
                DragValue::new(&mut self.sim.walk_sigma)
                    .prefix("Walk σ: ")
                    .clamp_range(0.0..=f32::INFINITY)
                    .speed(1e-5),
            );

            ui.separator();

            let mut rebuild_accel = false;
            rebuild_accel |= ui
                .add(
                    DragValue::new(&mut self.potential_cutoff)
                        .prefix("Potential cutoff: ")
                        .clamp_range(1e-6..=f32::INFINITY)
                        .speed(1e-2),
                )
                .changed();
            rebuild_accel |= ui
                .add(
                    DragValue::new(&mut self.editor_potential.attract)
                        .prefix("Attract: ")
                        .clamp_range(0.0..=f32::INFINITY)
                        .speed(1e-3),
                )
                .changed();
            rebuild_accel |= ui
                .add(
                    DragValue::new(&mut self.editor_potential.repulse)
                        .prefix("Repulse: ")
                        .clamp_range(0.0..=f32::INFINITY)
                        .speed(1e-4),
                )
                .changed();

            let radius = self.editor_potential.solve(self.potential_cutoff);
            ui.label(format!("Radius: {}", radius));

            if rebuild_accel {
                self.sim
                    .set_potential(self.editor_potential, self.potential_cutoff);
            }

            ui.horizontal(|ui| {
                let do_reset = ui.button("Reset").clicked();
                ui.add(DragValue::new(&mut self.n).prefix("# of particles: "));
                if do_reset {
                    self.sim = Sim::new(
                        self.n,
                        self.potential_cutoff,
                        Ruleset::default(),
                        self.sim.temperature,
                        self.sim.walk_sigma,
                    );
                }
            });
        });
    }

    fn update_sim(&mut self, io: &mut EngineIo, _query: &mut QueryResult) {
        for _ in 0..self.substeps {
            self.sim.step();
        }

        io.send(&UploadMesh {
            mesh: state_mesh(&self.sim.state, &self.sim.rules),
            id: PARTICLES_RDR,
        });
    }
}

fn state_mesh(state: &State, rules: &Ruleset) -> Mesh {
    let vertices = state
        .positions
        .iter()
        .zip(&state.types)
        .map(|(pos, ty)| Vertex::new([pos.x, 0., pos.y], rules.particles[*ty as usize].color))
        .collect();

    let indices = (0..state.positions.len() as u32).collect();
    Mesh { vertices, indices }
}

#[derive(Clone, Copy, Debug)]
struct LennardJones {
    /// Attractive coefficient
    pub attract: f32,
    /// Repulsive coefficient
    pub repulse: f32,
}

#[derive(Clone)]
struct State {
    positions: Vec<Vec2>,
    types: Vec<u8>,
}

struct Sim {
    state: State,
    accel: QueryAccelerator,
    rules: Ruleset,
    temperature: f32,
    walk_sigma: f32,
}

impl Sim {
    pub fn new(n: usize, cutoff: f32, rules: Ruleset, temperature: f32, walk_sigma: f32) -> Self {
        let mut rng = rng();

        let s = 0.1;

        let positions = (0..n)
            .map(|_| Vec2::new(rng.gen_range(-s..=s), rng.gen_range(-s..=s)))
            .collect();

        let rules = Ruleset::default();

        let types = (0..n)
            .map(|_| rng.gen_range(0..rules.particles.len() as u8))
            .collect();

        let state = State { positions, types };

        let radius = rules
            .particles
            .iter()
            .map(|rule| {
                rule.interactions
                    .iter()
                    .map(|r| r.solve(cutoff))
                    .max_by(|a, b| a.total_cmp(b))
                    .unwrap()
            })
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();

        cimvr_engine_interface::println!("Radius: {}", radius);

        let accel = QueryAccelerator::new(&state.positions, radius);

        Self {
            state,
            accel,
            temperature,
            walk_sigma,
            rules,
        }
    }

    pub fn set_potential(&mut self, potential: LennardJones, cutoff: f32) {
        let radius = potential.solve(cutoff);
        self.accel = QueryAccelerator::new(&self.state.positions, radius);
        //self.potential = potential;
        todo!();
    }

    pub fn step(&mut self) {
        let ref mut rng = rng();

        // Pick a particle
        let idx = rng.gen_range(0..self.state.positions.len());

        // Perterb it
        let original = self.state.positions[idx];
        let mut candidate = original;
        let normal = Normal::new(0.0, self.walk_sigma).unwrap();
        candidate.x += normal.sample(rng);
        candidate.y += normal.sample(rng);

        // Calculate the candidate change in energy
        let old_energy = self.energy_due_to(idx, original);
        let new_energy = self.energy_due_to(idx, candidate);
        let delta_e = new_energy - old_energy;

        // Decide whether to accept the change
        let probability = (-delta_e / self.temperature).exp();
        if probability > rng.gen_range(0.0..=1.0) {
            self.state.positions[idx] = candidate;
            self.accel.replace_point(idx, original, candidate);
        }
    }

    pub fn energy_due_to(&self, idx: usize, pos: Vec2) -> f32 {
        let mut energy = 0.;

        let my_ruleset = self.state.types[idx] as usize;
        let my_ruleset = &self.rules.particles[my_ruleset];

        for neighbor in self.accel.query_neighbors(&self.state.positions, idx, pos) {
            let distance = self.state.positions[neighbor].distance(pos);
            let potential = self.state.types[neighbor] as usize;
            let potential = my_ruleset.interactions[potential];

            let potential = potential.eval(distance);
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
        4. * ((self.repulse / radius).powi(12) - (self.attract / radius).powi(6))
    }

    /// Solve for the radius given a potential magnitude (sign is discarded)
    /// This is useful for finding an appropriate cutoff radius for local interactions
    /// https://www.desmos.com/calculator/itneqxndwy
    pub fn solve(&self, potential: f32) -> f32 {
        assert!(self.repulse >= 0.);
        assert!(self.attract >= 0.);

        if self.repulse < 0.5 {
            // Corner case where the solution is numerically inaccurate
            self.attract * (1. / potential).powf(1. / 6.)
        } else {
            let a = self.repulse.powi(12);
            let b = -self.attract.powi(6);
            let c = -potential;

            // The familiar formula
            let p = (-b - (b.powi(2) - 4. * a * c).sqrt()) / a / 2.;

            p.abs().powf(-1. / 6.)
        }
    }
}

impl Default for LennardJones {
    fn default() -> Self {
        Self {
            attract: 0.01,
            repulse: 0.007,
        }
    }
}

struct Ruleset {
    particles: Vec<ParticleRules>,
}

struct ParticleRules {
    color: [f32; 3],
    interactions: Vec<LennardJones>,
}

impl Default for Ruleset {
    fn default() -> Self {
        Self {
            particles: vec![
                ParticleRules {
                    color: [1., 0.1, 0.1],
                    interactions: vec![
                        LennardJones {
                            attract: 0.01,
                            repulse: 0.007,
                        },
                        LennardJones {
                            attract: 0.03,
                            repulse: 0.007,
                        },
                    ],
                },
                ParticleRules {
                    color: [0.1, 0.5, 1.0],
                    interactions: vec![
                        LennardJones {
                            attract: 0.01,
                            repulse: 0.07,
                        },
                        LennardJones {
                            attract: 0.01,
                            repulse: 0.007,
                        },
                    ],
                },
            ],
        }
    }
}
