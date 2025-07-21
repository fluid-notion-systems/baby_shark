use crate::algo::merge_points::IndexedVertices;
use nalgebra::SVector;
use simba::scalar::RealField;

/// A simple mesh representation that wraps IndexedVertices.
///
/// This struct provides a convenient interface for working with 3D meshes,
/// including conversions to and from other mesh representations.
#[derive(Debug, Clone)]
pub struct Mesh<S: RealField> {
    pub inner: IndexedVertices<3, S>,
}

impl<S: RealField> Mesh<S> {
    /// Create a new mesh from vertices and indices.
    pub fn new(vertices: Vec<SVector<S, 3>>, indices: Vec<usize>) -> Self {
        Self {
            inner: IndexedVertices {
                points: vertices,
                indices,
            },
        }
    }

    /// Get the vertices of the mesh.
    pub fn vertices(&self) -> &[SVector<S, 3>] {
        &self.inner.points
    }

    /// Get the indices of the mesh.
    pub fn indices(&self) -> &[usize] {
        &self.inner.indices
    }

    /// Get a mutable reference to the vertices.
    pub fn vertices_mut(&mut self) -> &mut Vec<SVector<S, 3>> {
        &mut self.inner.points
    }

    /// Get a mutable reference to the indices.
    pub fn indices_mut(&mut self) -> &mut Vec<usize> {
        &mut self.inner.indices
    }

    /// Get the number of vertices in the mesh.
    pub fn vertex_count(&self) -> usize {
        self.inner.points.len()
    }

    /// Get the number of indices in the mesh.
    pub fn index_count(&self) -> usize {
        self.inner.indices.len()
    }

    /// Get the number of triangles in the mesh (assumes triangular faces).
    pub fn triangle_count(&self) -> usize {
        self.inner.indices.len() / 3
    }
}

impl<S: RealField> From<IndexedVertices<3, S>> for Mesh<S> {
    fn from(indexed_vertices: IndexedVertices<3, S>) -> Self {
        Self {
            inner: indexed_vertices,
        }
    }
}

impl<S: RealField> From<Mesh<S>> for IndexedVertices<3, S> {
    fn from(mesh: Mesh<S>) -> Self {
        mesh.inner
    }
}

#[cfg(feature = "bevy")]
mod bevy_conversions {
    use super::*;
    use bevy_render::mesh::{Mesh as BevyMesh, PrimitiveTopology};
    use bevy_render::render_asset::RenderAssetUsages;

    impl From<BevyMesh> for Mesh<f32> {
        fn from(bevy_mesh: BevyMesh) -> Self {
            // Extract vertex positions
            let positions = bevy_mesh
                .attribute(BevyMesh::ATTRIBUTE_POSITION)
                .expect("Mesh must have position attribute")
                .as_float3()
                .expect("Position attribute must be float3")
                .to_vec();

            let vertices: Vec<SVector<f32, 3>> = positions
                .into_iter()
                .map(|[x, y, z]| SVector::<f32, 3>::new(x, y, z))
                .collect();

            // Extract indices
            let indices = if let Some(indices) = bevy_mesh.indices() {
                match indices {
                    bevy_render::mesh::Indices::U16(indices) => {
                        indices.iter().map(|&i| i as usize).collect()
                    }
                    bevy_render::mesh::Indices::U32(indices) => {
                        indices.iter().map(|&i| i as usize).collect()
                    }
                }
            } else {
                // If no indices, create a simple index buffer
                (0..vertices.len()).collect()
            };

            Mesh::new(vertices, indices)
        }
    }

    impl From<Mesh<f32>> for BevyMesh {
        fn from(mesh: Mesh<f32>) -> Self {
            let vertices = mesh.vertices();
            let indices = mesh.indices();

            // Convert vertices to flat array of positions
            let positions: Vec<[f32; 3]> = vertices.iter().map(|v| [v[0], v[1], v[2]]).collect();

            // Convert indices to u32
            let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();

            // Create Bevy mesh
            let mut bevy_mesh = BevyMesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            );

            // Set vertex positions
            bevy_mesh.insert_attribute(BevyMesh::ATTRIBUTE_POSITION, positions);

            // Set indices
            bevy_mesh.insert_indices(bevy_render::mesh::Indices::U32(indices_u32));

            // Compute normals if not present
            if bevy_mesh.attribute(BevyMesh::ATTRIBUTE_NORMAL).is_none() {
                // For flat normals, we need to duplicate vertices first
                bevy_mesh.duplicate_vertices();
                bevy_mesh.compute_flat_normals();
            }

            bevy_mesh
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_bevy_mesh_round_trip() {
            // Create a simple triangle mesh
            let vertices = vec![
                SVector::<f32, 3>::new(0.0, 0.0, 0.0),
                SVector::<f32, 3>::new(1.0, 0.0, 0.0),
                SVector::<f32, 3>::new(0.0, 1.0, 0.0),
            ];
            let indices = vec![0, 1, 2];

            let mesh = Mesh::new(vertices.clone(), indices.clone());

            // Convert to Bevy mesh and back
            let bevy_mesh: BevyMesh = mesh.clone().into();
            let mesh_back: Mesh<f32> = bevy_mesh.into();

            // Check vertices
            assert_eq!(mesh_back.vertex_count(), 3);
            for (i, vertex) in mesh_back.vertices().iter().enumerate() {
                assert_eq!(vertex[0], vertices[i][0]);
                assert_eq!(vertex[1], vertices[i][1]);
                assert_eq!(vertex[2], vertices[i][2]);
            }

            // Check indices
            assert_eq!(mesh_back.indices(), &indices);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_creation() {
        let vertices = vec![
            SVector::<f32, 3>::new(0.0, 0.0, 0.0),
            SVector::<f32, 3>::new(1.0, 0.0, 0.0),
            SVector::<f32, 3>::new(0.0, 1.0, 0.0),
        ];
        let indices = vec![0, 1, 2];

        let mesh = Mesh::new(vertices.clone(), indices.clone());

        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.index_count(), 3);
        assert_eq!(mesh.triangle_count(), 1);
        assert_eq!(mesh.vertices(), &vertices);
        assert_eq!(mesh.indices(), &indices);
    }

    #[test]
    fn test_from_indexed_vertices() {
        let indexed_vertices = IndexedVertices {
            points: vec![
                SVector::<f32, 3>::new(0.0, 0.0, 0.0),
                SVector::<f32, 3>::new(1.0, 0.0, 0.0),
                SVector::<f32, 3>::new(0.0, 1.0, 0.0),
            ],
            indices: vec![0, 1, 2],
        };

        let mesh: Mesh<f32> = indexed_vertices.clone().into();

        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.index_count(), 3);
        assert_eq!(mesh.vertices(), &indexed_vertices.points);
        assert_eq!(mesh.indices(), &indexed_vertices.indices);
    }

    #[test]
    fn test_into_indexed_vertices() {
        let vertices = vec![
            SVector::<f32, 3>::new(0.0, 0.0, 0.0),
            SVector::<f32, 3>::new(1.0, 0.0, 0.0),
            SVector::<f32, 3>::new(0.0, 1.0, 0.0),
        ];
        let indices = vec![0, 1, 2];

        let mesh = Mesh::new(vertices.clone(), indices.clone());
        let indexed_vertices: IndexedVertices<3, f32> = mesh.into();

        assert_eq!(indexed_vertices.points, vertices);
        assert_eq!(indexed_vertices.indices, indices);
    }
}
