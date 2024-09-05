/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref make_dense_object function.
 */

#ifndef OPENKALMAN_MAKE_DENSE_OBJECT_HPP
#define OPENKALMAN_MAKE_DENSE_OBJECT_HPP

namespace OpenKalman
{
  /**
   * \brief Make a default, dense, writable matrix based on a list of Dimensions describing the sizes of each index.
   * \details The result will be uninitialized.
   * \tparam T A dummy matrix or array from the relevant library (size, shape, and layout are ignored)
   * \tparam layout The \ref Layout of the resulting object. If this is Layout::none, it will be the default layout for the library of T.
   * \tparam Scalar The scalar type of the resulting object (by default, it is the same scalar type as T).
   * \param d a tuple of \ref vector_space_descriptor describing dimensions of each index.
   * Trailing 1D indices my be omitted.
   */
#ifdef __cpp_concepts
  template<indexible T, Layout layout = Layout::none, scalar_type Scalar = scalar_type_of_t<T>, vector_space_descriptor...D>
    requires (layout != Layout::stride) and interface::make_default_defined_for<T, layout, Scalar, D&&...>
  constexpr writable auto
#else
  template<typename T, Layout layout = Layout::none, typename Scalar = scalar_type_of_t<T>, typename...D, std::enable_if_t<
    indexible<T> and scalar_type<Scalar> and (vector_space_descriptor<D> and ...) and (layout != Layout::stride) and
    interface::make_default_defined_for<T, layout, Scalar, D&&...>, int> = 0>
  constexpr auto
#endif
  make_dense_object(D&&...d)
  {
    using Traits = interface::library_interface<std::decay_t<T>>;
    return Traits::template make_default<layout, Scalar>(std::forward<D>(d)...);
  }


} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_DENSE_OBJECT_HPP
