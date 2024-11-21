/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Forward declarations for object traits.
 */

#ifndef OPENKALMAN_TRAITS_HPP
#define OPENKALMAN_TRAITS_HPP

#include <type_traits>

// basic traits

#include "linear-algebra/traits/scalar_type_of.hpp"
#include "indexible.hpp"
#include "linear-algebra/traits/index_count.hpp"
#include "linear-algebra/traits/vector_space_descriptor_of.hpp"
#include "linear-algebra/traits/index_dimension_of.hpp"
#include "dynamic_dimension.hpp"
#include "linear-algebra/traits/dynamic_index_count.hpp"
#include "has_dynamic_dimensions.hpp"
#include "dimension_size_of_index_is.hpp"
#include "vector.hpp"
#include "linear-algebra/traits/max_tensor_order.hpp"

#include "has_untyped_index.hpp" // Is this necessary?
#include "all_fixed_indices_are_euclidean.hpp" // Is this necessary?

#include "wrappable.hpp"
#include "has_nested_object.hpp"
#include "linear-algebra/traits/nested_object_of.hpp"

// shape-based traits

#include "linear-algebra/traits/internal/not_more_fixed_than.hpp"
#include "linear-algebra/traits/internal/less_fixed_than.hpp"
#include "linear-algebra/traits/internal/maybe_same_shape_as_vector_space_descriptors.hpp"
#include "linear-algebra/traits/internal/has_uniform_static_vector_space_descriptors.hpp"

#include "compatible_with_vector_space_descriptor_collection.hpp"
#include "vector_space_descriptors_may_match_with.hpp"
#include "vector_space_descriptors_match_with.hpp"

#include "one_dimensional.hpp"
#include "square_shaped.hpp"
#include "empty_object.hpp"

// constants:

#include "linear-algebra/traits/internal/get_singular_component.hpp"
#include "linear-algebra/traits/constant_coefficient.hpp"
#include "constant_matrix.hpp"
#include "zero.hpp"
#include "linear-algebra/traits/constant_diagonal_coefficient.hpp"
#include "constant_diagonal_matrix.hpp"
#include "identity_matrix.hpp"

// special matrices:

#include "triangular_matrix.hpp"
#include "linear-algebra/traits/triangle_type_of.hpp"
#include "triangular_adapter.hpp"
#include "diagonal_matrix.hpp"
#include "diagonal_adapter.hpp"
#include "hermitian_matrix.hpp"
#include "hermitian_adapter.hpp"
#include "linear-algebra/traits/hermitian_adapter_type_of.hpp"

// other:

#include "writable.hpp"

#include "element_gettable.hpp" // deprecated
#include "index_range_for.hpp"
#include "writable_by_component.hpp"

#include "directly_accessible.hpp"
#include "linear-algebra/traits/layout_of.hpp"

#include "object-types.hpp"


#endif //OPENKALMAN_TRAITS_HPP
