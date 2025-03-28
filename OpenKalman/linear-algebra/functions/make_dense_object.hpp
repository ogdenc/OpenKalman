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
   * \brief Make a default, dense, writable matrix with a set of \ref coordinate::pattern objects defining the dimensions.
   * \details The result will be uninitialized.
   * \tparam T A dummy matrix or array from the relevant library (size, shape, and layout are ignored)
   * \tparam layout The \ref Layout of the resulting object. If this is Layout::none, it will be the default layout for the library of T.
   * \tparam Scalar The scalar type of the resulting object (by default, it is the same scalar type as T).
   * \param d a tuple of \ref coordinate::pattern describing dimensions of each index.
   * Trailing 1D indices my be omitted.
   */
#ifdef __cpp_concepts
  template<indexible T, Layout layout = Layout::none, value::number Scalar = scalar_type_of_t<T>, pattern_collection Descriptors>
    requires (layout != Layout::stride) and
    interface::make_default_defined_for<T, layout, Scalar, decltype(internal::to_euclidean_vector_space_descriptor_collection(std::declval<Descriptors&&>()))>
  constexpr writable auto
#else
  template<typename T, Layout layout = Layout::none, typename Scalar = scalar_type_of_t<T>, typename Descriptors, std::enable_if_t<
    indexible<T> and value::number<Scalar> and pattern_collection<D> and (layout != Layout::stride) and
    interface::make_default_defined_for<T, layout, Scalar, decltype(internal::to_euclidean_vector_space_descriptor_collection(std::declval<Descriptors&&>()))>, int> = 0>
  constexpr auto
#endif
  make_dense_object(Descriptors&& descriptors)
  {
    decltype(auto) d = internal::remove_trailing_1D_descriptors(std::forward<Descriptors>(descriptors));
    using D = decltype(d);
    using Traits = interface::library_interface<std::decay_t<T>>;
    if constexpr (coordinate::euclidean_pattern_collection<D>)
    {
      return Traits::template make_default<layout, Scalar>(std::forward<D>(d));
    }
    else
    {
      auto ed = internal::to_euclidean_vector_space_descriptor_collection(d);
      return make_vector_space_adapter(Traits::template make_default<layout, Scalar>(ed), std::forward<D>(d));
    }
  }


  /**
   * \overload
   * \brief \ref coordinate::pattern object are specified as parameters.
   */
#ifdef __cpp_concepts
  template<indexible T, Layout layout = Layout::none, value::number Scalar = scalar_type_of_t<T>, coordinate::pattern...Ds>
    requires (layout != Layout::stride) and
    interface::make_default_defined_for<T, layout, Scalar, decltype(std::tuple {get_size(std::declval<Ds&&>())...})>
  constexpr writable auto
#else
  template<typename T, Layout layout = Layout::none, typename Scalar = scalar_type_of_t<T>, typename...Ds, std::enable_if_t<
    indexible<T> and value::number<Scalar> and (... and coordinate::pattern<Ds>) and (layout != Layout::stride) and
    interface::make_default_defined_for<T, layout, Scalar, std::tuple<Ds&&...>>, int> = 0>
  constexpr auto
#endif
  make_dense_object(Ds&&...ds)
  {
    return make_dense_object<T, layout, Scalar>(std::tuple {std::forward<Ds>(ds)...});
  }


} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_DENSE_OBJECT_HPP
