/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref triangular_adapter.
 */

#ifndef OPENKALMAN_TRIANGULAR_ADAPTER_HPP
#define OPENKALMAN_TRIANGULAR_ADAPTER_HPP


namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_triangular_adapter : std::false_type {};

    template<typename T>
    struct is_triangular_adapter<T, std::enable_if_t<interface::indexible_object_traits<stdcompat::remove_cvref_t<T>>::is_triangular_adapter>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a type is a triangular adapter of triangle type tri.
   * \details A triangular adapter takes a matrix and presents a view in which, in one or both triangular
   * (or trapezoidal) sides on either side of the diagonal are zero. The matrix need not be square.
   * \tparam T A matrix or tensor.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept triangular_adapter = interface::indexible_object_traits<stdcompat::remove_cvref_t<T>>::is_triangular_adapter and
#else
  constexpr bool triangular_adapter = detail::is_triangular_adapter<T>::value and has_nested_object<T> and
#endif
    has_nested_object<T>;


}

#endif
