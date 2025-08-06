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
 * \brief Definition for \ref writable.
 */

#ifndef OPENKALMAN_WRITABLE_HPP
#define OPENKALMAN_WRITABLE_HPP


namespace OpenKalman
{
  /**
   * \internal
   * \brief Specifies that T is a dense, writable matrix.
   * \todo Add some assignability test?
   */
  template<typename T>
#ifdef __cpp_concepts
  concept writable =
    indexible<T> and interface::indexible_object_traits<std::decay_t<T>>::is_writable and
    (not std::is_const_v<std::remove_reference_t<T>>) and std::copy_constructible<std::decay_t<T>>;
#else
  constexpr bool writable =
    indexible<T> and interface::is_explicitly_writable<T>::value and (not std::is_const_v<std::remove_reference_t<T>>) and
    stdcompat::copy_constructible<std::decay_t<T>> and std::is_move_constructible_v<std::decay_t<T>>;
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_WRITABLE_HPP
