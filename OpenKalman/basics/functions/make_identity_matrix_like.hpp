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
 * \brief Definitions for \ref make_identity_matrix_like.
 */

#ifndef OPENKALMAN_MAKE_IDENTITY_MATRIX_LIKE_HPP
#define OPENKALMAN_MAKE_IDENTITY_MATRIX_LIKE_HPP

namespace OpenKalman
{
  namespace detail
  {
    template<typename T, typename Scalar, typename D0, typename D1, typename...Ds>
    constexpr auto
    make_identity_matrix_like_impl(D0&& d0, D1&& d1, Ds&&...ds)
    {
      if constexpr (interface::make_identity_matrix_defined_for<T, Scalar, D0&&, D1&&, Ds&&...>)
      {
        return interface::library_interface<std::decay_t<T>>::template make_identity_matrix<Scalar>(std::forward<D0>(d0), std::forward<D1>(d1), std::forward<Ds>(ds)...);
      }
      else if constexpr (fixed_vector_space_descriptor<D0> and fixed_vector_space_descriptor<D1>)
      {
        if constexpr (dimension_size_of_v<D0> == 1 and dimension_size_of_v<D1> == 1)
          return make_constant<T, Scalar, 1>(std::forward<D0>(d0), std::forward<D1>(d1), std::forward<Ds>(ds)...);
        else if constexpr (internal::prefix_of<D0, D1>)
          return make_diagonal_matrix(make_constant<T, Scalar, 1>(d0, std::forward<Ds>(ds)...), std::forward<D0>(d0), std::forward<D1>(d1));
        else
          return make_diagonal_matrix(make_constant<T, Scalar, 1>(d1, std::forward<Ds>(ds)...), std::forward<D0>(d0), std::forward<D1>(d1));
      }
      else
      {
        auto d = get_dimension_size_of(d0) < get_dimension_size_of(d1) ? DynamicDescriptor<Scalar> {d0} : DynamicDescriptor<Scalar> {d1};
        return make_diagonal_matrix(make_constant<T, Scalar, 1>(d, std::forward<Ds>(ds)...), std::forward<D0>(d0), std::forward<D1>(d1));
      }
    }

  } // namespace detail


  /**
   * \brief Make an identity matrix with a particular set of dimensions.
   * \tparam T Any matrix or tensor within the relevant library.
   * \tparam Scalar An optional scalar type for the new zero matrix. By default, T's scalar type is used.
   * \param Ds A set of \ref vector_space_descriptor items defining the dimensions of each index.
   */
#ifdef __cpp_concepts
  template<indexible T, scalar_type Scalar = scalar_type_of_t<T>, vector_space_descriptor D0, vector_space_descriptor D1, vector_space_descriptor...Ds>
  requires (not fixed_vector_space_descriptor<D0> or not fixed_vector_space_descriptor<D1> or internal::prefix_of<D0, D1> or internal::prefix_of<D1, D0>)
  constexpr identity_matrix auto
#else
  template<typename T, typename Scalar = typename scalar_type_of<T>::type, typename D0, typename D1, typename...Ds, std::enable_if_t<
    indexible<T> and scalar_type<Scalar> and vector_space_descriptor<D0> and (vector_space_descriptor<D1> and ... and vector_space_descriptor<Ds>) and
    (not fixed_vector_space_descriptor<D0> or not fixed_vector_space_descriptor<D1> or internal::prefix_of<D0, D1> or internal::prefix_of<D1, D0>), int> = 0>
  constexpr auto
#endif
  make_identity_matrix_like(D0&& d0 = Dimensions<1>{}, D1&& d1 = Dimensions<1>{}, Ds&&...ds)
  {
    return std::apply([](D0&& d0, D1&& d1, auto&&...ds){
      return detail::make_identity_matrix_like_impl<T, Scalar>(std::forward<D0>(d0), std::forward<D1>(d1), std::forward<decltype(ds)>(ds)...);
      }, std::tuple_cat(std::forward_as_tuple(std::forward<D0>(d0), std::forward<D1>(d1)), internal::remove_trailing_1D_descriptors(std::forward<Ds>(ds)...)));
  }


  /**
   * \overload
   * \brief Make an identity matrix with the same size and shape as an argument, specifying a new scalar type.
   * \tparam Arg The matrix or array on which the new identity matrix is patterned. It need not be square.
   * \tparam Scalar A scalar type for the new matrix.
   * \return An identity matrix of the same dimensions as Arg (even if not square).
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar, indexible Arg>
  constexpr identity_matrix auto
#else
  template<typename Scalar, typename Arg, std::enable_if_t<scalar_type<Scalar> and indexible<Arg>, int> = 0>
  constexpr auto
#endif
  make_identity_matrix_like(Arg&& arg)
  {
    if constexpr (identity_matrix<Arg> and std::is_same_v<Scalar, scalar_type_of_t<Arg>>) return std::forward<Arg>(arg);
    else return std::apply([](auto&&...ds){
      return make_identity_matrix_like<Arg, Scalar>(std::forward<decltype(ds)>(ds)...);
      }, all_vector_space_descriptors(std::forward<Arg>(arg)));
  }


  /**
   * \overload
   * \brief Make an identity matrix with the same size and shape as an argument.
   * \tparam Arg The matrix or array on which the new zero matrix is patterned.
   * \return An identity matrix of the same dimensions as Arg (even if not square).
   */
#ifdef __cpp_concepts
  template<indexible Arg>
  constexpr identity_matrix auto
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
  constexpr auto
#endif
  make_identity_matrix_like(Arg&& arg)
  {
    return make_identity_matrix_like<scalar_type_of_t<Arg>>(std::forward<Arg>(arg));
  }


} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_IDENTITY_MATRIX_LIKE_HPP
