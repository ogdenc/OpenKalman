/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref diagonal_of function.
 */

#ifndef OPENKALMAN_DIAGONAL_OF_HPP
#define OPENKALMAN_DIAGONAL_OF_HPP

#include "linear-algebra/concepts/constant_object.hpp"

namespace OpenKalman
{
  /**
   * \brief Extract a column vector (or column slice for rank>2 tensors) comprising the diagonal elements.
   * \tparam Arg An \ref indexible object, which can have any rank and may or may not be square
   * \returns A column vector or slice corresponding to the diagonal.
   */
#ifdef __cpp_concepts
  template<indexible Arg>
  constexpr indexible decltype(auto)
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  diagonal_of(Arg&& arg)
  {
    if constexpr (one_dimensional<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (interface::diagonal_of_defined_for<Arg&&>)
    {
      return interface::library_interface<stdex::remove_cvref_t<Arg>>::diagonal_of(std::forward<Arg>(arg));
    }
    else if constexpr (constant_object<Arg>)
    {


      auto ds = get_pattern_collection(std::forward<Arg>(arg));
      if constexpr (pattern_collection<decltype(ds)>)
      {
        return internal::make_constant_diagonal_from_descriptors<Arg>(
          constant_value {std::forward<Arg>(arg)},
          std::tuple_cat(ds, std::tuple{patterns::Axis{}, patterns::Axis{}}));
      }
      else
      {
        return internal::make_constant_diagonal_from_descriptors<Arg>(constant_value {std::forward<Arg>(arg)}, ds);
      }
    }
    else if constexpr (constant_diagonal_object<Arg>)
    {
      auto ds = get_pattern_collection(std::forward<Arg>(arg));
      if constexpr (pattern_collection<decltype(ds)>)
      {      
        return internal::make_constant_diagonal_from_descriptors<Arg>(
          constant_diagonal_value {std::forward<Arg>(arg)},
          std::tuple_cat(get_pattern_collection(std::forward<Arg>(arg)), std::tuple{patterns::Axis{}, patterns::Axis{}}));
      }
      else
      {
        return internal::make_constant_diagonal_from_descriptors<Arg>(constant_diagonal_value {std::forward<Arg>(arg)}, ds);
      }
    }
    else
    {
      return interface::library_interface<stdex::remove_cvref_t<Arg>>::diagonal_of(std::forward<Arg>(arg));
    }
  }


}

#endif
