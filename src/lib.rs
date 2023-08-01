use cimvr_common::{
    glam::Vec2,
    render::{Mesh, MeshHandle, Primitive, Render, UploadMesh, Vertex},
    ui::{GuiInputMessage, GuiTab},
    Transform,
};
use cimvr_engine_interface::{make_app_state, pcg::Pcg, pkg_namespace, prelude::*};
use rand::prelude::*;
use rand_distr::Normal;

struct ClientState {
    ui: GuiTab,
    sim: Sim,
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

        let sim = Sim::new(100);

        Self { ui, sim }
    }
}

impl ClientState {
    fn update_ui(&mut self, io: &mut EngineIo, _query: &mut QueryResult) {
        let energy = total_energy(&self.sim.current);

        //cimvr_engine_interface::println!("{}", energy);

        self.ui.show(io, |ui| {
            ui.label(format!("Total energy: {energy:.003}"));
        });
    }

    fn update_sim(&mut self, io: &mut EngineIo, _query: &mut QueryResult) {
        self.sim.step();
        io.send(&UploadMesh {
            mesh: state_mesh(&self.sim.current),
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

#[derive(Clone)]
struct State {
    positions: Vec<Vec2>,
}

struct Sim {
    current: State,
}

impl Sim {
    pub fn new(n: usize) -> Self {
        let mut rng = rng();

        let s = 0.1;

        let positions = (0..n)
            .map(|_| Vec2::new(rng.gen_range(-s..=s), rng.gen_range(-s..=s)))
            .collect();

        let current = State { positions };

        Self { current }
    }

    pub fn step(&mut self) {
        let ref mut rng = rng();
        let normal = Normal::new(0.0, 0.001).unwrap();

        let n_steps = 100;


        for _ in 0..n_steps {
            let mut new_state = self.current.clone();

            let part = new_state.positions.choose_mut(rng).unwrap();
            part.x += normal.sample(rng);
            part.y += normal.sample(rng);

            let old_energy = total_energy(&self.current);
            let new_energy = total_energy(&new_state);

            if new_energy < old_energy {
                self.current = new_state;
            }
        }
    }
}

fn rng() -> SmallRng {
    let u = ((Pcg::new().gen_u32() as u64) << 32) | Pcg::new().gen_u32() as u64;
    SmallRng::seed_from_u64(u)
}

fn potential(a: Vec2, b: Vec2) -> f32 {
    let r = a.distance(b);
    let epsilon = 0.01;
    let attract = 0.01;
    let repulse = 0.01;

    4. * epsilon * ((repulse / r).powi(12) - (attract / r).powi(6))
}

fn total_energy(state: &State) -> f32 {
    let mut sum = 0.;
    for i in 0..state.positions.len() {
        for j in (i + 1)..state.positions.len() {
            sum += potential(state.positions[i], state.positions[j]);
        }
    }
    sum
}
