/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref make_library_wrapper function.
 */

#ifndef OPENKALMAN_MAKE_LIBRARY_WRAPPER_HPP
#define OPENKALMAN_MAKE_LIBRARY_WRAPPER_HPP

namespace OpenKalman::internal
{
  /**
   * Make a \ref LibraryWrapper object.
   * \tparam Ps Parameters to be stored, if any
   */
#ifdef __cpp_concepts
  template<indexible Arg, indexible LibraryObject>
#else
  template<typename Arg, typename LibraryObject, std::enable_if_t<indexible<Arg> and indexible<LibraryObject>, int> = 0>
#endif
  inline auto
  make_library_wrapper(Arg&& arg)
  {
    return LibraryWrapper<Arg, LibraryObject> {std::forward<Arg>(arg)};
  }


  /**
   * \overload
   * \brief Create a self-contained, wrapped object of type T, using constructor arguments Ps...
   */
#ifdef __cpp_concepts
  template<indexible T, indexible LibraryObject, typename...Ps> requires std::constructible_from<T, std::add_lvalue_reference_t<Ps>...>
#else
  template<typename T, typename LibraryObject, typename...Ps, std::enable_if_t<indexible<T> and indexible<LibraryObject> and
    std::is_constructible_v<T, std::add_lvalue_reference_t<Ps>...>, int> = 0>
#endif
  inline auto
  make_library_wrapper(Ps&&...ps)
  {
    return LibraryWrapper<T, LibraryObject, std::remove_reference_t<Ps>...> {std::forward<Ps>(ps)...};
  }

} // namespace OpenKalman::internal


#endif //OPENKALMAN_MAKE_LIBRARY_WRAPPER_HPP
