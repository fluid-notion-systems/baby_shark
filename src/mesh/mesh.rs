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
    /// Creates a mesh from an iterator of f32 values, chunking them into groups of 3 for vertices.
    /// Ignores any remaining values if the count is not a multiple of 9 (3 vertices * 3 coordinates).
    ///
    /// # Arguments
    /// * `floats` - An iterator of f32 values representing flattened triangle vertices
    ///
    /// # Example
    /// ```
    /// use baby_shark::mesh::Mesh;
    ///
    /// let floats = vec![
    ///     // First triangle
    ///     0.0, 0.0, 0.0,  // vertex 1
    ///     1.0, 0.0, 0.0,  // vertex 2
    ///     0.0, 1.0, 0.0,  // vertex 3
    ///     // Second triangle
    ///     1.0, 0.0, 0.0,  // vertex 1 (duplicate)
    ///     1.0, 1.0, 0.0,  // vertex 2
    ///     0.0, 1.0, 0.0,  // vertex 3 (duplicate)
    /// ];
    ///
    /// let mesh = Mesh::<f32>::from_iter(floats.into_iter());
    /// ```
    pub fn from_iter<T>(vertices: impl Iterator<Item = T>) -> Mesh<S>
    where
        S: crate::geometry::traits::RealNumber,
        T: Into<S>,
    {
        // Collect values into a vector, converting to S
        let float_vec: Vec<S> = vertices.map(|v| v.into()).collect();

        // Convert to nalgebra Vector3, using chunks_exact to process every 3 floats
        let nalgebra_vertices: Vec<SVector<S, 3>> = float_vec
            .chunks_exact(3)
            .map(|chunk| SVector::<S, 3>::new(chunk[0], chunk[1], chunk[2]))
            .collect();

        // Use merge_points to deduplicate vertices
        let indexed = crate::algo::merge_points::merge_points(nalgebra_vertices.into_iter());

        indexed.into()
    }

    /// Creates a mesh from a Vec of f32 values using zero-copy conversion via bytemuck.
    /// This is more efficient than `from_iter` when you have a `Vec<f32>` with length
    /// that's a multiple of 3, as it avoids copying the data.
    ///
    /// # Arguments
    /// * `floats` - A Vec<f32> with length that must be a multiple of 3
    ///
    /// # Truncation
    /// If the vector length is not a multiple of 3, the extra elements at the end are ignored.
    ///
    /// # Example
    /// ```
    /// use baby_shark::mesh::Mesh;
    ///
    /// let floats = vec![
    ///     // First triangle
    ///     0.0, 0.0, 0.0,  // vertex 1
    ///     1.0, 0.0, 0.0,  // vertex 2
    ///     0.0, 1.0, 0.0,  // vertex 3
    ///     // Second triangle
    ///     1.0, 0.0, 0.0,  // vertex 1 (duplicate)
    ///     1.0, 1.0, 0.0,  // vertex 2
    ///     0.0, 1.0, 0.0,  // vertex 3 (duplicate)
    /// ];
    ///
    /// let mesh = Mesh::<f32>::from_vec_zerocopy(floats);
    /// ```
    #[cfg(feature = "bytemuck")]
    pub fn from_vec_zerocopy(floats: Vec<f32>) -> Mesh<f32>
    where
        f32: crate::geometry::traits::RealNumber,
    {
        use bytemuck::cast_slice;

        // Truncate to ensure the vector length is a multiple of 3
        let truncated_len = (floats.len() / 3) * 3;
        let truncated_floats = &floats[..truncated_len];

        // Zero-copy cast from &[f32] to &[SVector<f32, 3>]
        // This works because SVector<f32, 3> has the same memory layout as [f32; 3]
        let nalgebra_slice: &[SVector<f32, 3>] = cast_slice(truncated_floats);

        // Convert slice to Vec - this does copy, but the cast itself was zero-copy
        let nalgebra_vertices: Vec<SVector<f32, 3>> = nalgebra_slice.to_vec();

        // Use merge_points to deduplicate vertices
        let indexed = crate::algo::merge_points::merge_points(nalgebra_vertices.into_iter());

        indexed.into()
    }

    /// Creates a mesh from a slice using zero-copy conversion via bytemuck.
    /// This is more efficient than `from_iter` when you have a slice of f32 values,
    /// as it avoids individual element processing.
    ///
    /// # Arguments
    /// * `floats` - A slice of f32 values. Length will be truncated to a multiple of 3.
    ///
    /// # Truncation
    /// If the slice length is not a multiple of 3, the extra elements at the end are ignored.
    ///
    /// # Example
    /// ```
    /// use baby_shark::mesh::Mesh;
    ///
    /// let floats = &[
    ///     // First triangle
    ///     0.0, 0.0, 0.0,  // vertex 1
    ///     1.0, 0.0, 0.0,  // vertex 2
    ///     0.0, 1.0, 0.0,  // vertex 3
    /// ];
    ///
    /// let mesh = Mesh::<f32>::from_slice_zerocopy(floats);
    /// ```
    #[cfg(feature = "bytemuck")]
    pub fn from_slice_zerocopy(floats: &[f32]) -> Mesh<f32>
    where
        f32: crate::geometry::traits::RealNumber,
    {
        use bytemuck::cast_slice;

        // Truncate to ensure the slice length is a multiple of 3
        let truncated_len = (floats.len() / 3) * 3;
        let truncated_floats = &floats[..truncated_len];

        // Zero-copy cast from &[f32] to &[SVector<f32, 3>]
        // This works because SVector<f32, 3> has the same memory layout as [f32; 3]
        let nalgebra_slice: &[SVector<f32, 3>] = cast_slice(truncated_floats);

        // Convert slice to Vec - this does copy, but the cast itself was zero-copy
        let nalgebra_vertices: Vec<SVector<f32, 3>> = nalgebra_slice.to_vec();

        // Use merge_points to deduplicate vertices
        let indexed = crate::algo::merge_points::merge_points(nalgebra_vertices.into_iter());

        indexed.into()
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
    use bevy_asset::RenderAssetUsages;
    use bevy_mesh::{Mesh as BevyMesh, PrimitiveTopology};

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
                    bevy_mesh::Indices::U16(indices) => {
                        indices.iter().map(|&i| i as usize).collect()
                    }
                    bevy_mesh::Indices::U32(indices) => {
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

    //TODO: make generic
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
            bevy_mesh.insert_indices(bevy_mesh::Indices::U32(indices_u32));

            // Compute normals if not present
            if bevy_mesh.attribute(BevyMesh::ATTRIBUTE_NORMAL).is_none() {
                // For flat normals, we need to duplicate vertices first
                bevy_mesh.duplicate_vertices();
                bevy_mesh.compute_flat_normals();
            }

            if bevy_mesh.attribute(BevyMesh::ATTRIBUTE_UV_0).is_none() {
                let vertex_count = bevy_mesh
                    .attribute(BevyMesh::ATTRIBUTE_POSITION)
                    .map(|attr| match attr {
                        bevy_mesh::VertexAttributeValues::Float32x3(positions) => positions.len(),
                        _ => 0,
                    })
                    .unwrap_or(0);
                let simple_uvs = vec![[0.0, 0.0]; vertex_count];
                bevy_mesh.insert_attribute(BevyMesh::ATTRIBUTE_UV_0, simple_uvs);
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

    #[test]
    fn test_from_iter_simple_triangle() {
        let floats = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mesh = Mesh::<f32>::from_iter(floats.iter().cloned());

        assert_eq!(mesh.triangle_count(), 1);
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.index_count(), 3);
    }

    #[test]
    fn test_from_iter_with_duplicates() {
        // Two triangles sharing vertices
        let floats = vec![
            // First triangle
            0.0, 0.0, 0.0, // vertex 0
            1.0, 0.0, 0.0, // vertex 1
            0.0, 1.0, 0.0, // vertex 2
            // Second triangle (shares vertices 1 and 2)
            1.0, 0.0, 0.0, // vertex 1 (duplicate)
            1.0, 1.0, 0.0, // vertex 3
            0.0, 1.0, 0.0, // vertex 2 (duplicate)
        ];

        let mesh = Mesh::<f32>::from_iter(floats.iter().cloned());

        // Should have deduplicated vertices
        assert_eq!(mesh.triangle_count(), 2);
        assert!(mesh.vertex_count() < 6); // Less than 6 due to deduplication
        assert_eq!(mesh.index_count(), 6); // 2 triangles × 3 indices
    }

    #[test]
    fn test_from_iter_incomplete_data() {
        // 10 floats - only 9 should be used (3 complete triangles worth)
        let floats = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mesh = Mesh::<f32>::from_iter(floats.iter().cloned());

        assert_eq!(mesh.triangle_count(), 1); // Only one complete triangle
        assert_eq!(mesh.index_count(), 3);
    }

    #[test]
    fn test_from_iter_empty() {
        let floats: Vec<f32> = vec![];
        let mesh = Mesh::<f32>::from_iter(floats.iter().cloned());

        assert_eq!(mesh.triangle_count(), 0);
        assert_eq!(mesh.vertex_count(), 0);
        assert_eq!(mesh.index_count(), 0);
    }

    #[cfg(feature = "bytemuck")]
    #[test]
    fn test_from_vec_zerocopy_simple() {
        let floats = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mesh = Mesh::<f32>::from_vec_zerocopy(floats);

        assert_eq!(mesh.triangle_count(), 1);
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.index_count(), 3);
    }

    #[cfg(feature = "bytemuck")]
    #[test]
    fn test_from_vec_zerocopy_truncation() {
        // 10 floats - should use only first 9
        let floats = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mesh = Mesh::<f32>::from_vec_zerocopy(floats);

        assert_eq!(mesh.triangle_count(), 1); // Only one complete triangle
        assert_eq!(mesh.index_count(), 3);
    }

    #[cfg(feature = "bytemuck")]
    #[test]
    fn test_from_slice_zerocopy_simple() {
        let floats = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mesh = Mesh::<f32>::from_slice_zerocopy(&floats);

        assert_eq!(mesh.triangle_count(), 1);
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.index_count(), 3);
    }

    #[cfg(feature = "bytemuck")]
    #[test]
    fn test_from_slice_zerocopy_with_duplicates() {
        let floats = [
            // First triangle
            0.0, 0.0, 0.0, // vertex 0
            1.0, 0.0, 0.0, // vertex 1
            0.0, 1.0, 0.0, // vertex 2
            // Second triangle (shares vertices)
            1.0, 0.0, 0.0, // vertex 1 (duplicate)
            1.0, 1.0, 0.0, // vertex 3
            0.0, 1.0, 0.0, // vertex 2 (duplicate)
        ];

        let mesh = Mesh::<f32>::from_slice_zerocopy(&floats);

        // Should have deduplicated vertices
        assert_eq!(mesh.triangle_count(), 2);
        assert!(mesh.vertex_count() < 6); // Less than 6 due to deduplication
        assert_eq!(mesh.index_count(), 6); // 2 triangles × 3 indices
    }

    #[cfg(feature = "bytemuck")]
    #[test]
    fn test_zerocopy_methods_produce_same_result() {
        let floats = vec![
            0.0, 0.0, 0.0, // triangle 1
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, // triangle 2 (duplicate vertex)
            1.0, 1.0, 0.0, 0.0, 1.0, 0.0, // duplicate vertex
        ];

        let mesh_from_vec = Mesh::<f32>::from_vec_zerocopy(floats.clone());
        let mesh_from_slice = Mesh::<f32>::from_slice_zerocopy(&floats);

        assert_eq!(
            mesh_from_vec.triangle_count(),
            mesh_from_slice.triangle_count()
        );
        assert_eq!(mesh_from_vec.vertex_count(), mesh_from_slice.vertex_count());
        assert_eq!(mesh_from_vec.index_count(), mesh_from_slice.index_count());
    }
}
