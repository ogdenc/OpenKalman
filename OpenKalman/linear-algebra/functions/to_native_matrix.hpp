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
  /**
   * \brief If it isn't already, convert Arg to a native object in the library associated with LibraryObject.
   * \details The new object will be one that is fully treated as native by the library associated with LibraryObject and
   * that can be an input in any OpenKalman function associated with library LibraryObject.
   * \tparam LibraryObject Any indexible object from the library to which Arg is to be converted. It's shape or scalar type are irrelevant.
   * \tparam Arg The argument
   */
#ifdef __cpp_concepts
  template<indexible LibraryObject, indexible Arg>
  inline decltype(auto)
  to_native_matrix(Arg&& arg)
#else
  template<typename LibraryObject, typename Arg, std::enable_if_t<indexible<LibraryObject> and indexible<Arg>, int> = 0>
  inline decltype(auto)
  to_native_matrix(Arg&& arg)
#endif
  {
    if constexpr (interface::to_native_matrix_defined_for<LibraryObject, Arg>)
      return interface::library_interface<std::decay_t<LibraryObject>>::to_native_matrix(std::forward<Arg>(arg));
    else
      return internal::LibraryWrapper<Arg, LibraryObject> {std::forward<Arg>(arg)};
  }

} // namespace OpenKalman

#endif //OPENKALMAN_TO_NATIVE_MATRIX_HPP
