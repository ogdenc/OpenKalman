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
 * \brief Definition of \ref make_eigen_wrapper function.
 */

#ifndef OPENKALMAN_MAKE_EIGEN_WRAPPER_HPP
#define OPENKALMAN_MAKE_EIGEN_WRAPPER_HPP

namespace OpenKalman::Eigen3
{
  /**
   * Make a \ref LibraryWrapper for the Eigen library.
   * \tparam Ps Parameters to be stored, if any
   */
#ifdef __cpp_concepts
  template<indexible Arg>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
  inline auto
  make_eigen_wrapper(Arg&& arg)
  {
    using M = Eigen::Matrix<scalar_type_of_t<Arg>, 0, 0>;
    return internal::make_library_wrapper<Arg, M>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Create a self-contained, wrapped object of type T in the Eigen library, using constructor arguments Ps...
   */
#ifdef __cpp_concepts
  template<indexible T, typename...Ps> requires std::constructible_from<T, std::add_lvalue_reference_t<Ps>...>
#else
  template<typename T, typename...Ps, std::enable_if_t<
    indexible<T> and std::is_constructible_v<T, std::add_lvalue_reference_t<Ps>...>, int> = 0>
#endif
  inline auto
  make_eigen_wrapper(Ps&&...ps)
  {
    using M = Eigen::Matrix<scalar_type_of_t<T>, 0, 0>;
    return internal::make_library_wrapper<T, M>(std::forward<Ps>(ps)...);
  }

} // namespace OpenKalman::Eigen3


#endif //OPENKALMAN_MAKE_EIGEN_WRAPPER_HPP
