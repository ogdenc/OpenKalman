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
 * \brief Definition of \ref coordinates::replicate_view and \ref coordinates::views::replicate.
 */

#ifndef OPENKALMAN_COORDINATES_VIEWS_REPLICATE_HPP
#define OPENKALMAN_COORDINATES_VIEWS_REPLICATE_HPP

#include "collections/collections.hpp"

namespace OpenKalman::coordinates::views
{
  namespace detail
  {
    template<typename Factor>
    struct replicate_closure : stdcompat::ranges::range_adaptor_closure<replicate_closure<Factor>>
    {
      constexpr replicate_closure(Factor f) : factor_ {std::move(f)} {};

#ifdef __cpp_concepts
      template<viewable_collection R>
#else
      template<typename R, std::enable_if_t<viewable_collection<R>, int> = 0>
#endif
      constexpr auto
      operator() (R&& r) const
      {
        return replicate_view {all(std::forward<R>(r)), factor_};
      }

    private:
      Factor factor_;
    };


    struct replicate_adaptor
    {
#ifdef __cpp_concepts
      template<values::index Factor>
#else
      template<typename Factor, std::enable_if_t<values::index<Factor>, int> = 0>
#endif
      constexpr auto
      operator() (Factor factor) const
      {
        return replicate_closure<Factor> {std::move(factor)};
      }


#ifdef __cpp_concepts
      template<viewable_collection R, values::index Factor>
#else
      template<typename R, typename Factor, std::enable_if_t<viewable_collection<R> and values::index<Factor>, int> = 0>
#endif
      constexpr auto
      operator() (R&& r, Factor factor) const
      {
        return replicate_view {all(std::forward<R>(r)), std::move(factor)};
      }

    };

  }


  /**
   * \brief a std::ranges::range_adaptor_closure for a set of replicated \ref pattern objects.
   */
  inline constexpr detail::replicate_adaptor replicate;

}


#endif
