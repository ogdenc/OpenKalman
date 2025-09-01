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
 * \brief Definition for \ref scalar_type_of.
 */

#ifndef OPENKALMAN_SCALAR_TYPE_OF_HPP
#define OPENKALMAN_SCALAR_TYPE_OF_HPP


namespace OpenKalman
{
  /**
   * \brief Type scalar type (e.g., std::float, std::double, std::complex<double>) of a tensor.
   * \tparam T A tensor or other array.
   * \internal \sa interface::indexible_object_traits::scalar_type
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct scalar_type_of {};


  /**
   * \overload
   */
#ifdef __cpp_concepts
  template<interface::scalar_type_defined_for T>
  struct scalar_type_of<T>
#else
  template<typename T>
  struct scalar_type_of<T, std::enable_if_t<interface::scalar_type_defined_for<T>>>
#endif
  {
    using type = typename interface::indexible_object_traits<stdcompat::remove_cvref_t<T>>::scalar_type;
  };


  /**
   * \brief helper template for \ref scalar_type_of.
   */
  template<typename T>
  using scalar_type_of_t = typename scalar_type_of<T>::type;


}

#endif
