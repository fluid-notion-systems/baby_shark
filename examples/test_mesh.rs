use baby_shark::algo::merge_points::IndexedVertices;
use baby_shark::mesh::Mesh;
use nalgebra::Vector3;

fn main() {
    println!("Testing baby_shark Mesh implementation");

    // Create a simple triangle mesh
    let vertices = vec![
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
    ];

    let indices = vec![0, 1, 2];

    // Create mesh using constructor
    let mesh = Mesh::new(vertices.clone(), indices.clone());

    println!(
        "Created mesh with {} vertices and {} indices",
        mesh.vertex_count(),
        mesh.index_count()
    );

    println!("Triangle count: {}", mesh.triangle_count());

    // Test From<IndexedVertices>
    let indexed_vertices = IndexedVertices {
        points: vertices.clone(),
        indices: indices.clone(),
    };

    let mesh_from_indexed: Mesh<f32> = indexed_vertices.into();
    println!("Created mesh from IndexedVertices");

    // Test Into<IndexedVertices>
    let back_to_indexed: IndexedVertices<3, f32> = mesh.into();
    println!("Converted mesh back to IndexedVertices");

    assert_eq!(back_to_indexed.points.len(), 3);
    assert_eq!(back_to_indexed.indices.len(), 3);

    #[cfg(feature = "bevy")]
    {
        use bevy_render::mesh::{Mesh as BevyMesh, PrimitiveTopology};
        use bevy_render::render_asset::RenderAssetUsages;

        println!("\nTesting Bevy conversions...");

        // Create a new mesh for Bevy conversion
        let mesh_for_bevy = Mesh::new(vertices, indices);

        // Convert to Bevy mesh
        let bevy_mesh: BevyMesh = mesh_for_bevy.clone().into();
        println!("Converted to Bevy mesh");

        // Convert back from Bevy mesh
        let mesh_from_bevy: Mesh<f32> = bevy_mesh.into();
        println!("Converted back from Bevy mesh");

        assert_eq!(mesh_from_bevy.vertex_count(), 3);
        assert_eq!(mesh_from_bevy.index_count(), 3);
        println!("Bevy round-trip conversion successful!");
    }

    println!("\nAll tests passed!");
}
