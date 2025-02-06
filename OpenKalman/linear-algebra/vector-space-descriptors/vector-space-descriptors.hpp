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


/**
 * \internal
 * \brief The namespace for vector-space descriptor code.
 */
namespace OpenKalman::descriptor {}


// interfaces:

#include "interfaces/vector_space_traits.hpp"
#include "interfaces/coordinate_set_traits.hpp"
#include "interfaces/traits-defined.hpp"

// initial concepts:

#include "concepts/composite_vector_space_descriptor.hpp"
#include "concepts/atomic_vector_space_descriptor.hpp"
#include "concepts/vector_space_descriptor.hpp"

#include "concepts/vector_space_descriptor_tuple.hpp"
#include "concepts/vector_space_descriptor_range.hpp"
#include "concepts/vector_space_descriptor_collection.hpp"

// interface-based traits and functions

#include "functions/internal/get_component_collection.hpp"
#include "functions/get_component_count.hpp"
#include "traits/vector_space_component_count.hpp"

#include "concepts/static_vector_space_descriptor.hpp"
#include "concepts/static_vector_space_descriptor_tuple.hpp"
#include "concepts/static_vector_space_descriptor_collection.hpp"
#include "concepts/dynamic_vector_space_descriptor.hpp"

#include "functions/internal/get_component_start_indices.hpp"
#include "functions/internal/get_euclidean_component_start_indices.hpp"

#include "functions/internal/get_index_table.hpp"
#include "functions/get_size.hpp"
#include "traits/dimension_size_of.hpp"

#include "functions/internal/get_euclidean_index_table.hpp"
#include "functions/get_euclidean_size.hpp"
#include "traits/euclidean_dimension_size_of.hpp"

#include "functions/get_is_euclidean.hpp"
#include "concepts/euclidean_vector_space_descriptor.hpp"
#include "concepts/euclidean_vector_space_descriptor_tuple.hpp"
#include "concepts/euclidean_vector_space_descriptor_collection.hpp"

#include "functions/get_type_index.hpp"

#include "functions/to_euclidean_element.hpp"
#include "functions/from_euclidean_element.hpp"
#include "functions/get_wrapped_component.hpp"
#include "functions/set_wrapped_component.hpp"

#include "traits/scalar_type_of.hpp"

// other

#include "internal/forward-declarations.hpp"
#include "traits/static_concatenate.hpp"

#include "functions/internal/canonical_equivalent.hpp"
#include "functions/comparison-operators.hpp"
#include "concepts/maybe_equivalent_to.hpp"
#include "concepts/equivalent_to.hpp"

#include "descriptors/internal/Concatenate.hpp"
#include "descriptors/internal/Replicate.hpp"
#include "functions/arithmetic-operators.hpp"
#include "concepts/internal/maybe_prefix_of.hpp"
#include "concepts/internal/prefix_of.hpp"
#include "descriptors/internal/Slice.hpp"
#include "descriptors/internal/Reverse.hpp"

// descriptors:

#include "descriptors/StaticDescriptor.hpp"
#include "descriptors/Dimensions.hpp"
#include "interfaces/index.hpp"
#include "descriptors/DynamicDescriptor.hpp"

#include "descriptors/Distance.hpp"
#include "descriptors/Angle.hpp"
#include "descriptors/Inclination.hpp"
#include "descriptors/Polar.hpp"
#include "descriptors/Spherical.hpp"

// traits for manipulating static descriptors

#include "traits/internal/uniform_static_vector_space_descriptor_query.hpp"
#include "concepts/uniform_static_vector_space_descriptor.hpp"
#include "traits/uniform_static_vector_space_descriptor_component_of.hpp"
#include "concepts/equivalent_to_uniform_static_vector_space_descriptor_component_of.hpp"

// functions:

#include "functions/internal/is_uniform_component_of.hpp" //
#include "functions/internal/remove_trailing_1D_descriptors.hpp" //
#include "functions/internal/best_vector_space_descriptor.hpp" //
#include "functions/internal/smallest_vector_space_descriptor.hpp" //
#include "functions/internal/largest_vector_space_descriptor.hpp" //
#include "functions/get_slice.hpp"

#include "functions/internal/to_euclidean_vector_space_descriptor_collection.hpp" //


#endif //OPENKALMAN_VECTOR_TYPES_HPP
