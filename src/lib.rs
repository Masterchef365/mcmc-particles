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

        let sim = Sim::new(1_000);

        Self { ui, sim }
    }
}

impl ClientState {
    fn update_ui(&mut self, io: &mut EngineIo, _query: &mut QueryResult) {
        self.ui.show(io, |ui| {
            if ui.button("Fuck yeah").clicked() {
                cimvr_engine_interface::println!("I love egui!");
            }
        });
    }

    fn update_sim(&mut self, io: &mut EngineIo, _query: &mut QueryResult) {
        self.sim.step();
        io.send(&UploadMesh {
            mesh: particles_mesh(&self.sim),
            id: PARTICLES_RDR,
        });
    }
}

fn particles_mesh(sim: &Sim) -> Mesh {
    let vertices = sim
        .positions
        .iter()
        .map(|p| Vertex::new([p.x, 0., p.y], [1.; 3]))
        .collect();
    let indices = (0..sim.positions.len() as u32).collect();
    Mesh { vertices, indices }
}

struct Sim {
    positions: Vec<Vec2>,
}

impl Sim {
    pub fn new(n: usize) -> Self {
        let mut rng = rng();

        let positions = (0..n)
            .map(|_| Vec2::new(rng.gen_range(-1.0..=1.0), rng.gen_range(-1.0..=1.0)))
            .collect();

        Self { positions }
    }

    pub fn step(&mut self) {
        let ref mut rng = rng();
        let normal = Normal::new(0.0, 0.001).unwrap();

        for _ in 0..self.positions.len() {
            let part = self.positions.choose_mut(rng).unwrap();
            part.x += normal.sample(rng);
            part.y += normal.sample(rng);
        }
    }
}

fn rng() -> SmallRng {
    let u = ((Pcg::new().gen_u32() as u64) << 32) | Pcg::new().gen_u32() as u64;
    SmallRng::seed_from_u64(u)
}
