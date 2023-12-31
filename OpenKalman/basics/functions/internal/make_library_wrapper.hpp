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

} // namespace OpenKalman::internal


#endif //OPENKALMAN_MAKE_LIBRARY_WRAPPER_HPP
