/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Overloaded general rank-update functions.
 */

#ifndef OPENKALMAN_RANK_UPDATE_HPP
#define OPENKALMAN_RANK_UPDATE_HPP

namespace OpenKalman
{
  using namespace interface;

  /**
   * \brief Do a rank update on a hermitian matrix.
   * \note This may (or may not) be performed as an in-place operation if argument A is writable.
   * \details The update is A += αUU<sup>*</sup>, returning the updated hermitian A.
   * If A is an lvalue reference and is writable, it will be updated in place and the return value will be an
   * lvalue reference to the same, updated A. Otherwise, the function returns a new matrix.
   * \tparam A The matrix to be rank updated.
   * \tparam U The update vector or matrix.
   * \returns an updated native, writable matrix in hermitian form.
   */
#ifdef __cpp_concepts
  template<hermitian_matrix<Likelihood::maybe> A, indexible U> requires
    (dynamic_dimension<U, 0> or dynamic_dimension<A, 0> or index_dimension_of_v<U, 0> == index_dimension_of_v<A, 0>) and
    std::convertible_to<scalar_type_of_t<U>, const scalar_type_of_t<A>>
  inline /*hermitian_matrix<Likelihood::maybe>*/ decltype(auto)
#else
  template<typename A, typename U, std::enable_if_t<indexible<U> and hermitian_matrix<A, Likelihood::maybe> and
    (dynamic_dimension<U, 0> or dynamic_dimension<A, 0> or index_dimension_of<U, 0>::value == index_dimension_of<A, 0>::value) and
    std::is_convertible_v<typename scalar_type_of<U>::type, const typename scalar_type_of<A>::type>, int> = 0>
  inline decltype(auto)
#endif
  rank_update_self_adjoint(A&& a, U&& u, scalar_type_of_t<A> alpha = 1)
  {
    if constexpr (dynamic_rows<A> or dynamic_rows<U>) if (get_index_dimension_of<0>(a) != get_index_dimension_of<0>(u))
      throw std::invalid_argument {
        "In rank_update_self_adjoint, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
        ") do not match rows of u (" + std::to_string(get_index_dimension_of<0>(u)) + ")"};

    if constexpr (not hermitian_matrix<A> and has_dynamic_dimensions<A>) if (get_index_dimension_of<0>(a) != get_index_dimension_of<1>(a))
      throw std::invalid_argument {
        "In rank_update_self_adjoint, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
        ") do not match columns of a (" + std::to_string(get_index_dimension_of<1>(a)) + ")"};

    constexpr auto t = hermitian_adapter_type_of_v<A>;

    if constexpr (zero_matrix<U>)
    {
      return std::forward<A>(a);
    }
    else if constexpr (one_by_one_matrix<A> or index_dimension_of_v<U, 0> == 1)
    {
      auto e = trace(a) + alpha * trace(u) * trace(conjugate(u));

      if constexpr (element_settable<A, std::size_t>)
        return set_element(a, e, 0);
      else if constexpr (element_settable<A, std::size_t, std::size_t>)
        return set_element(a, e, 0, 0);
      else
      {
        auto ret = make_dense_writable_matrix_from<A>(std::tuple{Dimensions<1>{}, Dimensions<1>{}}, e);
        if constexpr (std::is_assignable_v<A, decltype(std::move(ret))>) return a = std::move(ret);
        else return ret;
      }
    }
    else if constexpr (zero_matrix<A> and diagonal_matrix<U>)
    {
      return alpha * Cholesky_square(std::forward<U>(u));
    }
    else if constexpr (diagonal_matrix<A> and diagonal_matrix<U>)
    {
      auto d = sum(std::forward<A>(a), alpha * Cholesky_square(std::forward<U>(u)));
      if constexpr (std::is_assignable_v<A, decltype(std::move(d))>) return a = std::move(d);
      else return d;
    }
    else
    {
      using Trait = interface::LinearAlgebra<std::decay_t<A>>;
      return Trait::template rank_update_self_adjoint<t>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
  }


  /**
   * \brief Do a rank update on triangular matrix.
   * \note This may (or may not) be performed as an in-place operation if argument A is writable.
   * \details
   * - If A is lower-triangular, diagonal, or one-by-one, the update is AA<sup>*</sup> += αUU<sup>*</sup>,
   * returning the updated A.
   * - If A is upper-triangular, the update is A<sup>*</sup>A += αUU<sup>*</sup>, returning the updated A.
   * - If A is an lvalue reference and is writable, it will be updated in place and the return value will be an
   * lvalue reference to the same, updated A. Otherwise, the function returns a new matrix.
   * \tparam A The matrix to be rank updated.
   * \tparam U The update vector or matrix.
   * \returns an updated native, writable matrix in triangular (or diagonal) form.
   */
# ifdef __cpp_concepts
  template<triangular_matrix<Likelihood::maybe> A, indexible U> requires
    (dynamic_rows<A> or dynamic_rows<U> or row_dimension_of_v<A> == row_dimension_of_v<U>) and
    std::convertible_to<scalar_type_of_t<U>, const scalar_type_of_t<A>>
  inline /*triangular_matrix<Likelihood::maybe>*/ decltype(auto)
# else
  template<typename A, typename U, std::enable_if_t<triangular_matrix<A, Likelihood::maybe> and indexible<U> and
    (dynamic_rows<A> or dynamic_rows<U> or row_dimension_of<A>::value == row_dimension_of<U>::value) and
    std::is_convertible_v<scalar_type_of_t<U>, const scalar_type_of_t<A>>, int> = 0>
  inline decltype(auto)
# endif
  rank_update_triangular(A&& a, U&& u, scalar_type_of_t<A> alpha = 1)
  {
    if constexpr (dynamic_rows<A> or dynamic_rows<U>) if (get_index_dimension_of<0>(a) != get_index_dimension_of<0>(u))
      throw std::invalid_argument {
        "In rank_update_triangular, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
        ") do not match rows of u (" + std::to_string(get_index_dimension_of<0>(u)) + ")"};

    if constexpr (not triangular_matrix<A> and has_dynamic_dimensions<A>) if (get_index_dimension_of<0>(a) != get_index_dimension_of<1>(a))
      throw std::invalid_argument {
        "In rank_update_triangular, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
        ") do not match columns of a (" + std::to_string(get_index_dimension_of<1>(a)) + ")"};

    constexpr auto t = triangle_type_of_v<A>;
    if constexpr (zero_matrix<U>)
    {
      return std::forward<A>(a);
    }
    else if constexpr (one_by_one_matrix<A> or index_dimension_of_v<U, 0> == 1)
    {
      // Both A is known at compile time to be a 1-by-1 matrix.
      auto e = square_root(trace(a) * trace(conjugate(a)) + alpha * trace(u) * trace(conjugate(u)));

      if constexpr (element_settable<A, std::size_t>)
        return set_element(a, e, 0);
      else if constexpr (element_settable<A, std::size_t, std::size_t>)
        return set_element(a, e, 0, 0);
      else
      {
        auto ret = make_dense_writable_matrix_from<A>(std::tuple{Dimensions<1>{}, Dimensions<1>{}}, e);
        if constexpr (std::is_assignable_v<A, decltype(std::move(ret))>) return a = std::move(ret);
        else return ret;
      }
    }
    else if constexpr (zero_matrix<A>)
    {
      if constexpr (diagonal_matrix<U>)
      {
        return to_diagonal(square_root(alpha) * diagonal_of(std::forward<U>(u)));
      }
      else if constexpr (t == TriangleType::upper)
      {
        return QR_decomposition(square_root(alpha) * adjoint(std::forward<U>(u)));
      }
      else
      {
        return LQ_decomposition(square_root(alpha) * std::forward<U>(u));
      }
    }
    else if constexpr (diagonal_matrix<A> and diagonal_matrix<U>)
    {
      auto d = Cholesky_factor(sum(Cholesky_square(std::forward<A>(a)), alpha * Cholesky_square(std::forward<U>(u))));
      if constexpr (std::is_assignable_v<A, decltype(std::move(d))>) return a = std::move(d);
      else return d;
    }
    else
    {
      using Trait = interface::LinearAlgebra<std::decay_t<A>>;
      return Trait::template rank_update_triangular<t>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_RANK_UPDATE_HPP
