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
 * \brief Definition for \ref hermitian_adapter_concept.
 */

#ifndef OPENKALMAN_HERMITIAN_ADAPTER_HPP
#define OPENKALMAN_HERMITIAN_ADAPTER_HPP

#include "linear-algebra/traits/hermitian_adapter_type_of.hpp"

namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct hermitian_adapter_impl : std::false_type {};

    template<typename T>
    struct hermitian_adapter_impl<T, std::enable_if_t<
      interface::object_traits<stdex::remove_cvref_t<T>>::hermitian_adapter_type != triangle_type::none>> : std::true_type {};
  };
#endif


  /**
   * \brief Specifies that a type is a hermitian matrix adapter.
   * \details A hermitian adapter is also a \ref hermitian_matrix.
   * \tparam T A matrix or tensor.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept hermitian_adapter_concept =
    hermitian_adapter_type_of_v<T> != triangle_type::none;
#else
  constexpr bool hermitian_adapter_concept =
    hermitian_matrix<T> and
    detail::hermitian_adapter_impl<std::decay_t<T>, t>::value;
#endif


}

#endif
