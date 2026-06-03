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
 * \brief Definition for \ref hermitian_adapter_type_of.
 */

#ifndef OPENKALMAN_HERMITIAN_ADAPTER_TYPE_OF_HPP
#define OPENKALMAN_HERMITIAN_ADAPTER_TYPE_OF_HPP

#include "linear-algebra/enumerations.hpp"
#include "linear-algebra/concepts/hermitian_matrix.hpp"

namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename = T, typename = void, typename...Ts>
    struct hermitian_adapter_type_of_impl : std::integral_constant<triangle_type, triangle_type::none> {};

    template<typename T, typename...Ts>
    struct hermitian_adapter_type_of_impl<
      T,
      std::enable_if_t<(hermitian_matrix<T> and ... and hermitian_matrix<Ts>) and
        (... and (interface::object_traits<stdex::remove_cvref_t<T>>::hermitian_adapter_type ==
          interface::object_traits<stdex::remove_cvref_t<Ts>>::hermitian_adapter_type))>,
      Ts...>
      : std::integral_constant<triangle_type, interface::object_traits<stdex::remove_cvref_t<T>>::hermitian_adapter_type> {};
  };
#endif


  /**
   * \brief The triangle_type associated with the storage triangle of one or more matrices.
   * \details If there is no common triangle type, the result is triangle_type::none.
   * If the matrices have a dynamic shape, the result assumes the matrices are square.
   */
#ifdef __cpp_concepts
  template<typename T, typename...Ts>
  struct hermitian_adapter_type_of : std::integral_constant<triangle_type, triangle_type::none> {};


  /**
   * \overload
   */
  template<typename T, typename...Ts> requires
    (hermitian_matrix<T> and ... and hermitian_matrix<Ts>) and
    (... and (interface::object_traits<stdex::remove_cvref_t<T>>::hermitian_adapter_type ==
      interface::object_traits<stdex::remove_cvref_t<Ts>>::hermitian_adapter_type))
  struct hermitian_adapter_type_of<T, Ts...>
    : std::integral_constant<triangle_type, interface::object_traits<stdex::remove_cvref_t<T>>::hermitian_adapter_type> {};
#else
  template<typename T, typename...Ts>
  struct hermitian_adapter_type_of<T, Ts...> : detail::hermitian_adapter_type_of_impl<T, void, Ts...> {};
#endif


  /**
   * \brief The triangle_type associated with the storage triangle of a \ref hermitian_matrix.
   * \details Possible values are \ref triangle_type::lower "lower", \ref triangle_type::upper "upper", or
   * \ref triangle_type::any "any".
   */
  template<typename T, typename...Ts>
  constexpr auto hermitian_adapter_type_of_v = hermitian_adapter_type_of<T, Ts...>::value;


}

#endif
