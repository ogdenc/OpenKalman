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
 * \brief Definitions relating to \ref vector_space_descriptor object.
 *
 * \dir vector-space-descriptor/descriptors
 * \brief Files defining \ref vector_space_descriptor objects.
 *
 * \dir vector-space-descriptor/concepts
 * \brief Concepts relating to \ref vector_space_descriptor objects.
 *
 * \dir vector-space-descriptor/functions
 * \brief Files for functions relating to \ref vector_space_descriptor objects.
 *
 * \dir vector-space-descriptor/traits
 * \brief Traits relating to \ref vector_space_descriptor objects.
 *
 * \dir vector-space-descriptor/internal
 * \internal
 * \brief Internal files relating to \ref vector_space_descriptor objects.
 *
 * \dir vector-space-descriptor/tests
 * \internal
 * \brief Tests relating to \ref vector_space_descriptor objects.
 *
 * \file
 * \brief Comprehensive header file including all classes and definitions relating to a \ref vector_space_descriptor
 */

#ifndef OPENKALMAN_VECTOR_TYPES_HPP
#define OPENKALMAN_VECTOR_TYPES_HPP


// interfaces:

#include "linear-algebra/vector-space-descriptors/interfaces/static_vector_space_descriptor_traits.hpp" //
#include "linear-algebra/vector-space-descriptors/interfaces/dynamic_vector_space_descriptor_traits.hpp" //

// initial concepts:

#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/dynamic_vector_space_descriptor.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp" //

#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor_tuple.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor_collection.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor_tuple.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor_collection.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor_tuple.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor_collection.hpp" //

#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp" //

#include "linear-algebra/vector-space-descriptors/concepts/composite_vector_space_descriptor.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/atomic_static_vector_space_descriptor.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/maybe_equivalent_to.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/equivalent_to.hpp" //

// interface-based traits and functions

#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/euclidean_dimension_size_of.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/vector_space_component_count.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/dimension_difference_of.hpp" //

#include "linear-algebra/vector-space-descriptors/functions/get_dimension_size_of.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/get_euclidean_dimension_size_of.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/get_vector_space_descriptor_component_count_of.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/get_vector_space_descriptor_is_euclidean.hpp" //

#include "linear-algebra/vector-space-descriptors/functions/to_euclidean_element.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/from_euclidean_element.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/get_wrapped_component.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/set_wrapped_component.hpp" //

// internal traits:

#include "linear-algebra/vector-space-descriptors/traits/internal/prefix_base_of.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/internal/prefix_of.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/internal/suffix_of.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/internal/suffix_base_of.hpp" //

// descriptors:

#include "linear-algebra/vector-space-descriptors/descriptors/Dimensions.hpp" //
#include "linear-algebra/vector-space-descriptors/descriptors/StaticDescriptor.hpp" //
#include "linear-algebra/vector-space-descriptors/descriptors/DynamicDescriptor.hpp" //

#include "linear-algebra/vector-space-descriptors/descriptors/Distance.hpp" //
#include "linear-algebra/vector-space-descriptors/descriptors/Angle.hpp" //
#include "linear-algebra/vector-space-descriptors/descriptors/Inclination.hpp" //
#include "linear-algebra/vector-space-descriptors/descriptors/Polar.hpp" //
#include "linear-algebra/vector-space-descriptors/descriptors/Spherical.hpp" //

// traits for manipulating static descriptors

#include "linear-algebra/vector-space-descriptors/traits/replicate_static_vector_space_descriptor.hpp" //

#include "linear-algebra/vector-space-descriptors/traits/internal/canonical_static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/traits/concatenate_static_vector_space_descriptor.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/reverse_static_vector_space_descriptor.hpp"

#include "linear-algebra/vector-space-descriptors/traits/internal/uniform_static_vector_space_descriptor_query.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/uniform_static_vector_space_descriptor.hpp" //
#include "linear-algebra/vector-space-descriptors/traits/uniform_static_vector_space_descriptor_component_of.hpp" //
#include "linear-algebra/vector-space-descriptors/concepts/equivalent_to_uniform_static_vector_space_descriptor_component_of.hpp" //

// collection traits

#include "linear-algebra/vector-space-descriptors/traits/internal/vector_space_descriptor_collection_common_type.hpp" //

// functions:

#include "linear-algebra/vector-space-descriptors/functions/comparison-operators.hpp"

#include "linear-algebra/vector-space-descriptors/functions/internal/replicate_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/is_uniform_component_of.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/remove_trailing_1D_descriptors.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/best_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/smallest_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/largest_vector_space_descriptor.hpp"

#include "linear-algebra/vector-space-descriptors/functions/internal/split_head_tail.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/internal/static_vector_space_descriptor_slice.hpp" //
#include "linear-algebra/vector-space-descriptors/functions/get_vector_space_descriptor_slice.hpp" //

#include "linear-algebra/vector-space-descriptors/functions/internal/to_euclidean_vector_space_descriptor_collection.hpp"


#endif //OPENKALMAN_VECTOR_TYPES_HPP
