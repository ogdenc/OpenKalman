/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref compares_with_pattern_collection.
 */

#ifndef OPENKALMAN_COMPARES_WITH_PATTERN_COLLECTION_HPP
#define OPENKALMAN_COMPARES_WITH_PATTERN_COLLECTION_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/traits/pattern_collection_type_of.hpp"

namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename P, auto comp, applicability a, typename = void>
    struct compares_with_pattern_collection_impl : std::false_type {};

    template<typename T, typename P, auto comp, applicability a>
    struct compares_with_pattern_collection_impl<T, P, comp, a, std::enable_if_t<
      patterns::collection_compares_with<typename pattern_collection_type_of<T>::type, P, comp, a>>>
        : std::true_type {};
  }
#endif


  /**
   * \brief Compares the associated pattern collection of \ref indexible T with \ref pattern_collection D.
   * \tparam T An \ref indexible object
   * \tparam P A \ref pattern_collection
   */
  template<typename T, typename P, auto comp = &stdex::is_eq, applicability a = applicability::permitted>
#ifdef __cpp_concepts
  concept compares_with_pattern_collection =
    indexible<T> and
    patterns::collection_compares_with<pattern_collection_type_of_t<T>, P, comp, a>;
#else
  constexpr bool compares_with_pattern_collection =
    detail::compares_with_pattern_collection_impl<T, P, comp, a>::value;
#endif

}

#endif
