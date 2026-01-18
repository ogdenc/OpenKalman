/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2026 Christopher Lee Ogden <ogden@gatech.edu>
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

#ifndef OPENKALMAN_LINEAR_ALGEBRA_HPP
#define OPENKALMAN_LINEAR_ALGEBRA_HPP

#include "patterns/patterns.hpp"

#include "enumerations.hpp"

// interfaces

#include "interfaces/object_traits.hpp"
#include "interfaces/library_interface.hpp"
#include "interfaces/interfaces-defined.hpp"

// mdspan-derived traits

#include "concepts/indexible.hpp"
#include "traits/get_mdspan.hpp"

#include "traits/count_indices.hpp"
#include "traits/index_count.hpp"

#include "traits/element_type_of.hpp"

#include "traits/layout_of.hpp"

// extents and patterns

#include "traits/get_pattern_collection.hpp"
#include "traits/pattern_collection_type_of.hpp"

#include "traits/get_index_pattern.hpp"
#include "traits/get_index_extent.hpp"
#include "traits/index_dimension_of.hpp"

#include "traits/dynamic_index_count.hpp"
#include "concepts/dynamic_dimension.hpp"
#include "concepts/has_dynamic_dimensions.hpp"

#include "traits/tensor_order.hpp"
#include "traits/max_tensor_order.hpp"

#include "concepts/dimension_size_of_index_is.hpp"

// shapes

#include "traits/patterns_match.hpp"
#include "concepts/patterns_may_match_with.hpp"
#include "concepts/patterns_match_with.hpp"
#include "concepts/compares_with_pattern_collection.hpp"
#include "concepts/pattern_collection_for.hpp"

#include "concepts/square_shaped.hpp"
#include "traits/is_square_shaped.hpp"

#include "concepts/one_dimensional.hpp"
#include "traits/is_one_dimensional.hpp"

#include "concepts/empty_object.hpp"

#include "traits/is_vector.hpp"
#include "concepts/vector.hpp"

// indices and access

#include "concepts/index_collection_for.hpp"
#include "traits/access.hpp"
#include "traits/access_at.hpp"

// special matrices

#include "concepts/zero.hpp"
#include "traits/triangle_type_of.hpp"
#include "concepts/triangular_matrix.hpp"
#include "concepts/diagonal_matrix.hpp"
#include "concepts/hermitian_matrix.hpp"
//#include "traits/hermitian_adapter_type_of.hpp"

// constants

#include "concepts/constant_object.hpp"
#include "concepts/constant_diagonal_object.hpp"
#include "traits/constant_value.hpp"
#include "traits/constant_value_of.hpp"
#include "concepts/identity_matrix.hpp"

// linear algebra functions:

#include "concepts/copyable_from.hpp"
#include "functions/copy_from.hpp"

#include "adapters/pattern_adapter.hpp"
#include "functions/attach_patterns.hpp"

#include "functions/make_constant.hpp"
#include "functions/make_zero.hpp"

#include "functions/to_diagonal.hpp"
#include "functions/make_constant_diagonal.hpp"
#include "functions/make_identity_matrix.hpp"
#include "functions/diagonal_of.hpp"

#include "functions/conjugate.hpp"
#include "functions/transpose.hpp"
#include "functions/conjugate_transpose.hpp"

/*

#include "functions/internal/make_fixed_size_adapter.hpp"
#include "functions/internal/make_fixed_size_adapter_like.hpp"
#include "functions/internal/make_fixed_square_adapter_like.hpp"

#include "functions/internal/may_hold_components.hpp"
#include "functions/fill_components.hpp"


#include "functions/make_dense_object.hpp"
#include "functions/to_dense_object.hpp"
#include "functions/make_dense_object_from.hpp"

#include "functions/make_diagonal_adapter.hpp"

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
*/

// adapters:

#include "concepts/internal/not_more_fixed_than.hpp"
#include "concepts/internal/less_fixed_than.hpp"
#include "concepts/pattern_collection_for.hpp"

#include "concepts/has_untyped_index.hpp" // Is this necessary?
#include "concepts/all_fixed_indices_are_euclidean.hpp" // Is this necessary?
#include "concepts/wrappable.hpp"
#include "traits/get_wrappable.hpp"

#include "adapters/internal/forward-class-declarations.hpp"

#include "traits/nested_object.hpp"
#include "concepts/has_nested_object.hpp"
#include "traits/nested_object_of.hpp"

#include "concepts/triangular_adapter.hpp"
#include "concepts/internal/has_nested_vector.hpp"
#include "concepts/hermitian_adapter.hpp"

#include "concepts/object-types.hpp"

#include "adapters/adapters.hpp"


#include "linear-algebra/interfaces/stl/mdspan-object.hpp"
#include "linear-algebra/interfaces/stl/mdspan-library.hpp"
#include "linear-algebra/interfaces/stl/array-object.hpp"


#endif
