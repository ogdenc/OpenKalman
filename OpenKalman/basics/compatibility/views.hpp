/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Header file including all defined compatibility views equivalent to those in std::ranges::views.
 *
 * \dir views
 * \internal
 * \brief Compatibility definitions equivalent to standard-library views (std::ranges::views).
 */

#ifndef OPENKALMAN_COMPATIBILITY_VIEWS_HPP
#define OPENKALMAN_COMPATIBILITY_VIEWS_HPP

#include "ranges.hpp"

#ifndef __cpp_lib_ranges
#include "views/view_interface.hpp"
#include "views/view-concepts.hpp"
#include "views/range_adaptor_closure.hpp"
#include "views/ref_view.hpp"
#include "views/owning_view.hpp"
#include "views/all.hpp"
#include "views/empty.hpp"
#include "views/single.hpp"
#include "views/iota.hpp"
#include "views/transform.hpp"
#include "views/reverse.hpp"
#endif

#ifndef __cpp_lib_ranges_concat
#include "views/concat.hpp"
#endif

#ifndef __cpp_lib_ranges_to_container
#include "views/to.hpp"
#endif

#endif //OPENKALMAN_COMPATIBILITY_VIEWS_HPP
