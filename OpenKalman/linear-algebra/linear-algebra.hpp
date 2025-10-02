/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \dir
 * \brief Definitions and interface for basic linear algebra.
 *
 * \file
 * \brief Basic forward definitions for OpenKalman as a whole.
 * \details This should be included by any OpenKalman file, including interface files.
 */

/**
 */


#ifndef OPENKALMAN_LINEAR_ALGEBRA_HPP
#define OPENKALMAN_LINEAR_ALGEBRA_HPP

#include "coordinates/coordinates.hpp"

#include "enumerations.hpp"

// interfaces

#include "interfaces/default/indexible_object_traits.hpp"
#include "interfaces/object-traits-defined.hpp"
#include "interfaces/default/library_interface.hpp"
#include "interfaces/library-interfaces-defined.hpp"
#include "traits/internal/library_base.hpp"

// indices

#include "concepts/indexible.hpp"
#include "traits/count_indices.hpp"
#include "traits/index_count.hpp"

// mdspan characteristics

#include "traits/get_mdspan.hpp"
#include "traits/element_type_of.hpp"
#include "concepts/internal/layout_mapping_policy.hpp"
#include "traits/layout_of.hpp"
#include "concepts/writable.hpp"

// extents and associated patterns

#include "traits/get_pattern_collection.hpp"
#include "traits/get_index_pattern.hpp"
#include "traits/get_index_extent.hpp"
#include "traits/index_dimension_of.hpp"
#include "traits/tensor_order.hpp"

#include "traits/vector_space_descriptors_match.hpp"
#include "traits/is_square_shaped.hpp"
#include "traits/is_one_dimensional.hpp"
#include "traits/is_vector.hpp"
#include "traits/get_wrappable.hpp"

#include "traits/internal/truncate_indices.hpp"
#include "functions/get_component.hpp"


// basic

#include "concepts/dynamic_dimension.hpp"
#include "concepts/has_dynamic_dimensions.hpp"
#include "concepts/dimension_size_of_index_is.hpp"
#include "concepts/vector.hpp"

#include "concepts/has_untyped_index.hpp" // Is this necessary?
#include "concepts/all_fixed_indices_are_euclidean.hpp" // Is this necessary?

#include "concepts/wrappable.hpp"
#include "concepts/has_nested_object.hpp"

// shape-based

#include "concepts/internal/not_more_fixed_than.hpp"
#include "concepts/internal/less_fixed_than.hpp"
#include "concepts/internal/maybe_same_shape_as_vector_space_descriptors.hpp"

#include "concepts/compatible_with_vector_space_descriptor_collection.hpp"
#include "concepts/vector_space_descriptors_may_match_with.hpp"
#include "concepts/vector_space_descriptors_match_with.hpp"

#include "concepts/one_dimensional.hpp"
#include "concepts/square_shaped.hpp"
#include "concepts/empty_object.hpp"

// constants:

#include "concepts/constant_matrix.hpp"
#include "concepts/zero.hpp"
#include "concepts/constant_diagonal_matrix.hpp"
#include "concepts/identity_matrix.hpp"

// special matrices:

#include "concepts/triangular_matrix.hpp"
#include "concepts/triangular_adapter.hpp"
#include "concepts/diagonal_matrix.hpp"
#include "concepts/internal/has_nested_vector.hpp"
#include "concepts/hermitian_matrix.hpp"
#include "concepts/hermitian_adapter.hpp"

// other:


#include "concepts/element_gettable.hpp" // deprecated
#include "concepts/index_collection_for.hpp"

#include "concepts/directly_accessible.hpp"

#include "concepts/object-types.hpp"


// basic traits

#include "traits/dynamic_index_count.hpp"
#include "traits/max_tensor_order.hpp"

#include "traits/nested_object_of.hpp"

// constants:

#include "traits/constant_coefficient.hpp"
#include "traits/constant_diagonal_coefficient.hpp"

// special matrices:

#include "traits/triangle_type_of.hpp"
#include "traits/hermitian_adapter_type_of.hpp"




#include "adapters/internal/forward-class-declarations.hpp"



#include "functions/nested_object.hpp"

#include "functions/internal/make_fixed_size_adapter.hpp"
#include "functions/internal/make_fixed_size_adapter_like.hpp"
#include "functions/internal/make_fixed_square_adapter_like.hpp"

#include "functions/get_component.hpp"
#include "functions/set_component.hpp"

#include "functions/internal/may_hold_components.hpp"
#include "functions/fill_components.hpp"

#include "functions/to_native_matrix.hpp"
#include "functions/assign.hpp"
#include "functions/internal/assignable.hpp"

#include "functions/make_vector_space_adapter.hpp"

#include "functions/make_dense_object.hpp"
#include "functions/to_dense_object.hpp"
#include "functions/make_dense_object_from.hpp"

#include "functions/make_constant.hpp"
#include "functions/make_zero.hpp"
#include "functions/make_diagonal_adapter.hpp"

#include "functions/internal/make_constant_diagonal_from_descriptors.hpp"
#include "functions/make_identity_matrix_like.hpp"

#include "functions/transpose.hpp"

#include "functions/to_diagonal.hpp"
#include "functions/diagonal_of.hpp"

#include "functions/conjugate.hpp"
#include "functions/adjoint.hpp"

#include "functions/make_triangular_matrix.hpp"
#include "functions/make_hermitian_matrix.hpp"

#include "functions/internal/to_covariance_nestable.hpp"

#include "functions/to_euclidean.hpp"
#include "functions/from_euclidean.hpp"
#include "functions/wrap_angles.hpp"

#include "functions/broadcast.hpp"
#include "functions/n_ary_operation.hpp"
#include "functions/randomize.hpp"

#include "functions/scalar_product.hpp"
#include "functions/scalar_quotient.hpp"

#include "functions/internal/get_reduced_vector_space_descriptor.hpp"
#include "functions/internal/count_reduced_dimensions.hpp"
#include "functions/reduce.hpp"
#include "functions/average_reduce.hpp"

#include "functions/internal/check_block_limits.hpp"
#include "functions/get_slice.hpp"
#include "functions/set_slice.hpp"
#include "functions/get_chip.hpp"
#include "functions/set_chip.hpp"
#include "functions/internal/set_triangle.hpp"
#include "functions/internal/clip_square_shaped.hpp"

#include "functions/tile.hpp"
#include "functions/concatenate.hpp"
#include "functions/split.hpp"
#include "functions/chipwise_operation.hpp"

#include "functions/determinant.hpp"
#include "functions/trace.hpp"
#include "functions/sum.hpp"

#include "functions/contract.hpp"
#include "functions/contract_in_place.hpp"

#include "functions/LQ_decomposition.hpp"
#include "functions/QR_decomposition.hpp"

#include "functions/cholesky_square.hpp"
#include "functions/cholesky_factor.hpp"

#include "functions/internal/make_writable_square_matrix.hpp"
#include "functions/rank_update_hermitian.hpp"
#include "functions/rank_update_triangular.hpp"

#include "functions/solve.hpp"



#include "adapters/adapters.hpp"


#endif
