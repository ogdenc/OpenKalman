/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief definition for \ref to_native_matrix.
 */

#ifndef OPENKALMAN_TO_NATIVE_MATRIX_HPP
#define OPENKALMAN_TO_NATIVE_MATRIX_HPP

namespace OpenKalman
{
  using namespace interface;

  /**
   * \brief If it isn't already, convert Arg to a native matrix in library T.
   * \details The new matrix will be one in which basic matrix operations are defined.
   * \tparam T A matrix from the library to which Arg is to be converted.
   * \tparam Arg The argument
   */
#ifdef __cpp_concepts
  template<indexible T, indexible Arg> requires (not std::same_as<T, Arg>)
  inline decltype(auto)
  to_native_matrix(Arg&& arg)
#else
  template<typename T, typename Arg, std::enable_if_t<indexible<T> and indexible<Arg> and not std::is_same<T, Arg>::value, int> = 0>
  inline decltype(auto)
  to_native_matrix(Arg&& arg)
#endif
  {
    return EquivalentDenseWritableMatrix<std::decay_t<T>>::to_native_matrix(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief If it isn't already, convert arg into a native matrix within its library.
   */
#ifdef __cpp_concepts
  inline decltype(auto)
  to_native_matrix(indexible auto&& arg)
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
  inline decltype(auto)
  to_native_matrix(Arg&& arg)
#endif
  {
    return EquivalentDenseWritableMatrix<std::decay_t<decltype(arg)>>::to_native_matrix(std::forward<decltype(arg)>(arg));
  }


} // namespace OpenKalman

#endif //OPENKALMAN_TO_NATIVE_MATRIX_HPP
