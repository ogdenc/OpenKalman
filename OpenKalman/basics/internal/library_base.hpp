/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition of library_base.
 */

#ifndef OPENKALMAN_LIBRARY_BASE_HPP
#define OPENKALMAN_LIBRARY_BASE_HPP


namespace OpenKalman::internal
{
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
  template<typename Derived, typename PatternMatrix>
#else
  template<typename Derived, typename PatternMatrix, typename>
#endif
  struct library_base : MatrixTraits<std::decay_t<PatternMatrix>>::template MatrixBaseFrom<Derived>
  {};

} // namespace OpenKalman::internal

#endif //OPENKALMAN_LIBRARY_BASE_HPP
