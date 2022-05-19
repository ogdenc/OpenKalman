/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \dir
 * \brief Definitions relating to index descriptor types
 *
 * \dir index-descriptors/details
 * \brief Files implementing details regarding index descriptor types.
 *
 * \dir index-descriptors/tests
 * \brief Test files for coefficient types.
 *
 * \file
 * \brief Comprehensive header file including all index-type-related classes and definitions
 */

#ifndef OPENKALMAN_INDEX_DESCRIPTORS_HPP
#define OPENKALMAN_INDEX_DESCRIPTORS_HPP

#include "index-descriptor-interface-traits.hpp"
#include "index-descriptor-forward-declarations.hpp"

#include "Dimensions.hpp"
#include "Distance.hpp"
#include "Angle.hpp"
#include "Inclination.hpp"
#include "Polar.hpp"
#include "Spherical.hpp"

#include "TypedIndex.hpp"

#include "details/AbstractDynamicTypedIndexDescriptor.hpp"
#include "details/DynamicTypedIndexDescriptor.hpp"
#include "DynamicTypedIndex.hpp"

#include "index-descriptor-traits.hpp"
#include "index-descriptor-functions.hpp"

#endif //OPENKALMAN_INDEX_DESCRIPTORS_HPP
