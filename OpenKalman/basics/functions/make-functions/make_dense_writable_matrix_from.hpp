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
 * \brief Overloaded general functions for making dense writable objects.
 */

#ifndef OPENKALMAN_MAKE_DENSE_WRITABLE_MATRIX_FROM_HPP
#define OPENKALMAN_MAKE_DENSE_WRITABLE_MATRIX_FROM_HPP

namespace OpenKalman
{
  /**
   * \brief Convert the argument to a dense, writable matrix.
   * \tparam Scalar The Scalar type of the new matrix, if different than that of Arg
   * \tparam Arg The object from which the new matrix is based
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar, indexible Arg>
  constexpr /*writable*/ decltype(auto)
#else
  template<typename Scalar, typename Arg, std::enable_if_t<scalar_type<Scalar> and indexible<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  make_dense_writable_matrix_from(Arg&& arg)
  {
    if constexpr (writable<Arg> and std::is_same_v<Scalar, scalar_type_of_t<Arg>>)
      return std::forward<Arg>(arg);
    else
      return interface::EquivalentDenseWritableMatrix<std::decay_t<Arg>, std::decay_t<Scalar>>::convert(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Convert the argument to a dense, writable matrix.
   * \tparam Arg The object from which the new matrix is based
   */
#ifdef __cpp_concepts
  template<indexible Arg>
  constexpr /*writable*/ decltype(auto)
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  make_dense_writable_matrix_from(Arg&& arg)
  {
    if constexpr (writable<Arg>)
      return std::forward<Arg>(arg);
    else
      return interface::EquivalentDenseWritableMatrix<std::decay_t<Arg>, scalar_type_of_t<Arg>>::convert(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Create a dense, writable matrix with size and shape based on M, filled with a set of scalar components
   * \tparam M The matrix or array on which the new matrix is patterned.
   * \tparam Scalar An optional scalar type for the new matrix. By default, M's scalar type is used.
   * \tparam Ds Index descriptors describing the size of the resulting object.
   * \param d_tup A tuple of index descriptors Ds
   * \param args Scalar values to fill the new matrix.
   */
#ifdef __cpp_concepts
  template<indexible M, scalar_type Scalar = scalar_type_of_t<M>, index_descriptor...Ds, std::convertible_to<const Scalar> ... Args>
  requires (sizeof...(Args) % ((dynamic_index_descriptor<Ds> ? 1 : dimension_size_of_v<Ds>) * ... * 1) == 0)
  inline writable auto
#else
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wdiv-by-zero"
  template<typename M, typename Scalar = scalar_type_of_t<M>, typename...Ds, typename...Args, std::enable_if_t<
    indexible<M> and scalar_type<Scalar> and (index_descriptor<Ds> and ...) and
    (std::is_convertible_v<Args, const Scalar> and ...) and
    (sizeof...(Args) % ((dynamic_index_descriptor<Ds> ? 1 : dimension_size_of_v<Ds>) * ... * 1) == 0), int> = 0>
  inline auto
#endif
  make_dense_writable_matrix_from(const std::tuple<Ds...>& d_tup, Args...args)
  {
    using Trait = interface::EquivalentDenseWritableMatrix<std::decay_t<M>, std::decay_t<Scalar>>;
    using Nat = decltype(Trait::make_default(std::declval<Ds&&>()...));
    return MatrixTraits<std::decay_t<Nat>>::make(static_cast<const Scalar>(args)...);
  }
#ifndef __cpp_concepts
# pragma GCC diagnostic pop
#endif


  namespace detail
  {
    template<typename M, std::size_t...I>
    constexpr auto count_fixed_dims(std::index_sequence<I...>)
    {
      return ((dynamic_dimension<M, I> ? 1 : index_dimension_of_v<M, I>) * ... * 1);
    }


    template<typename M, std::size_t N>
    constexpr auto check_make_dense_args()
    {
      constexpr auto dims = count_fixed_dims<M>(std::make_index_sequence<max_indices_of_v<M>> {});
      return (N % dims == 0) and number_of_dynamic_indices_v<M> <= 1;
    }


    template<typename M, std::size_t dims, typename Scalar, std::size_t...I, typename...Args>
    inline auto make_dense_writable_matrix_from_impl(std::index_sequence<I...>, Args...args)
    {
      return make_dense_writable_matrix_from<M, Scalar>(
        std::tuple {[]{
          if constexpr (dynamic_dimension<M, I>) return Dimensions<sizeof...(Args) / dims>{};
          else return index_descriptor_of_t<M, I> {};
        }()...}, args...);
    }

  } // namespace detail


  /**
   * \overload
   * \brief Create a dense, writable matrix with size and shape based on M, filled with a set of scalar components
   * \tparam M The matrix or array on which the new matrix is patterned.
   * \tparam rows An optional row dimension for the new matrix. By default, M's row dimension is used.
   * \tparam columns An optional column dimension for the new matrix. By default, M's column dimension is used.
   * \tparam Scalar An optional scalar type for the new matrix. By default, M's scalar type is used.
   * \param args Scalar values to fill the new matrix.
   */
#ifdef __cpp_concepts
  template<indexible M, scalar_type Scalar = scalar_type_of_t<M>, std::convertible_to<const Scalar> ... Args>
  requires (detail::check_make_dense_args<M, sizeof...(Args)>())
  inline writable auto
#else
  template<typename M, typename Scalar = scalar_type_of_t<M>, typename ... Args, std::enable_if_t<
    indexible<M> and scalar_type<Scalar> and (std::is_convertible_v<Args, const Scalar> and ...) and
    (detail::check_make_dense_args<M, sizeof...(Args)>()), int> = 0>
  inline auto
#endif
  make_dense_writable_matrix_from(Args...args)
  {
    constexpr std::make_index_sequence<max_indices_of_v<M>> seq;
    constexpr auto dims = detail::count_fixed_dims<M>(seq);
    return detail::make_dense_writable_matrix_from_impl<M, dims, Scalar>(seq, args...);
  }


} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_DENSE_WRITABLE_MATRIX_FROM_HPP
