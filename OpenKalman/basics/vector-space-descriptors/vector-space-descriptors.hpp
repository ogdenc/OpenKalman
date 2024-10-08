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
 * \brief Definitions relating to \ref vector_space_descriptor.
 *
 * \dir vector-space-descriptor/details
 * \brief Files implementing details regarding \ref vector_space_descriptor objects.
 *
 * \dir vector-space-descriptor/tests
 * \brief Test files for coefficient types.
 *
 * \file
 * \brief Comprehensive header file including all classes and definitions relating to a \ref vector_space_descriptor
 */

#ifndef OPENKALMAN_VECTOR_TYPES_HPP
#define OPENKALMAN_VECTOR_TYPES_HPP

// interfaces:

#include "interfaces/fixed_vector_space_descriptor_traits.hpp"
#include "interfaces/dynamic_vector_space_descriptor_traits.hpp"

// forward definitions:

#include "traits/fixed_vector_space_descriptor.hpp"
#include "traits/dynamic_vector_space_descriptor.hpp"
#include "traits/vector_space_descriptor.hpp"

#include "traits/dimension_size_of.hpp"
#include "traits/euclidean_dimension_size_of.hpp"
#include "traits/vector_space_component_count.hpp"
#include "traits/dimension_difference_of.hpp"
#include "traits/euclidean_vector_space_descriptor.hpp"

#include "traits/descriptor-forward-traits.hpp"

#include "functions/get_dimension_size_of.hpp"
#include "functions/get_euclidean_dimension_size_of.hpp"
#include "functions/get_vector_space_descriptor_component_count_of.hpp"
#include "functions/get_vector_space_descriptor_is_euclidean.hpp"

#include "functions/to_euclidean_element.hpp"
#include "functions/from_euclidean_element.hpp"
#include "functions/get_wrapped_component.hpp"
#include "functions/set_wrapped_component.hpp"

// objects:
#include "interfaces/integral_interfaces.hpp"

#include "descriptors/Dimensions.hpp"
#include "descriptors/Distance.hpp"
#include "descriptors/Angle.hpp"
#include "descriptors/Inclination.hpp"
#include "descriptors/Polar.hpp"
#include "descriptors/Spherical.hpp"

#include "descriptors/FixedDescriptor.hpp"

#include "descriptors/details/AnyAtomicVectorSpaceDescriptor.hpp"
#include "descriptors/DynamicDescriptor.hpp"

// traits:

#include "traits/composite_vector_space_descriptor.hpp"
#include "traits/atomic_fixed_vector_space_descriptor.hpp"

#include "traits/concatenate_fixed_vector_space_descriptor.hpp"
#include "traits/replicate_fixed_vector_space_descriptor.hpp"
#include "traits/canonical_fixed_vector_space_descriptor.hpp"
#include "traits/reverse_fixed_vector_space_descriptor.hpp"

#include "traits/maybe_equivalent_to.hpp"
#include "traits/equivalent_to.hpp"

#include "traits/internal/is_prefix.hpp"
#include "traits/internal/prefix_of.hpp"
#include "traits/internal/suffix_of.hpp"
#include "traits/internal/base_of.hpp"
#include "traits/internal/split_head_tail_fixed.hpp"
#include "traits/internal/fixed_vector_space_descriptor_slice.hpp"

#include "traits/internal/uniform_fixed_vector_space_descriptor_query.hpp"
#include "traits/uniform_fixed_vector_space_descriptor.hpp"
#include "traits/uniform_fixed_vector_space_descriptor_component_of.hpp"
#include "traits/equivalent_to_uniform_fixed_vector_space_descriptor_component_of.hpp"

// functions:

#include "functions/comparison-operators.hpp"
#include "functions/arithmetic-operators.hpp"

#include "functions/internal/replicate_vector_space_descriptor.hpp"
#include "functions/internal/is_uniform_component_of.hpp"
#include "functions/internal/remove_trailing_1D_descriptors.hpp"
#include "functions/internal/best_vector_space_descriptor.hpp"
#include "functions/internal/smallest_vector_space_descriptor.hpp"
#include "functions/internal/largest_vector_space_descriptor.hpp"

#include "functions/get_vector_space_descriptor_slice.hpp"


#endif //OPENKALMAN_VECTOR_TYPES_HPP
