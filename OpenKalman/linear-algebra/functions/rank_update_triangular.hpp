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
 * \brief Definition for rank_update_triangular function.
 */

#ifndef OPENKALMAN_RANK_UPDATE_TRIANGULAR_HPP
#define OPENKALMAN_RANK_UPDATE_TRIANGULAR_HPP


namespace OpenKalman
{
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
  template<triangular_matrix<TriangleType::any> A, indexible U> requires
    dimension_size_of_index_is<U, 0, index_dimension_of_v<A, 0>, Applicability::permitted> and
    dimension_size_of_index_is<U, 0, index_dimension_of_v<A, 1>, Applicability::permitted> and
    std::convertible_to<scalar_type_of_t<U>, const scalar_type_of_t<A>>
  inline triangular_matrix<triangle_type_of_v<A> == TriangleType::upper ? TriangleType::upper : TriangleType::lower> decltype(auto)
# else
  template<typename A, typename U, std::enable_if_t<triangular_matrix<A, TriangleType::any> and indexible<U> and
    dimension_size_of_index_is<U, 0, index_dimension_of<A, 0>::value, Applicability::permitted> and
    dimension_size_of_index_is<U, 0, index_dimension_of<A, 1>::value, Applicability::permitted> and
    std::is_convertible_v<scalar_type_of_t<U>, const scalar_type_of_t<A>>, int> = 0>
  inline decltype(auto)
# endif
  rank_update_triangular(A&& a, U&& u, scalar_type_of_t<A> alpha = 1)
  {
    constexpr auto t = triangle_type_of_v<A> == TriangleType::upper ? TriangleType::upper : TriangleType::lower;

    if constexpr (zero<U>)
    {
      if constexpr (dynamic_dimension<A, 0> or dynamic_dimension<U, 0>) if (get_index_dimension_of<0>(a) != get_index_dimension_of<0>(u))
        throw std::invalid_argument {"In rank_update_triangular, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
          ") do not match rows of u (" + std::to_string(get_index_dimension_of<0>(u)) + ")"};

      return make_triangular_matrix<t>(std::forward<A>(a));
    }
    else if constexpr (dimension_size_of_index_is<A, 0, 1> or dimension_size_of_index_is<A, 1, 1> or dimension_size_of_index_is<U, 0, 1>)
    {
      if constexpr (dynamic_dimension<A, 0> or dynamic_dimension<U, 0>) if (get_index_dimension_of<0>(a) != get_index_dimension_of<0>(u))
        throw std::invalid_argument {"In rank_update_triangular, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
          ") do not match rows of u (" + std::to_string(get_index_dimension_of<0>(u)) + ")"};

      // From here on, A is known to be a 1-by-1 matrix.

      auto e = [](auto ax, const auto& uterm) {
          if constexpr (values::complex<scalar_type_of<A>>) return values::sqrt(ax * values::conj(ax) + uterm);
          else return values::sqrt(ax * ax + uterm);
      }(internal::get_singular_component(a), alpha * internal::get_singular_component(contract(u, adjoint(u))));

      if constexpr (writable_by_component<A&&>)
      {
        set_component(a, e, 0, 0);
        return make_triangular_matrix<t>(std::forward<A>(a));
      }
      else
      {
        auto ret {make_dense_object_from<A>(std::tuple{coordinates::Axis{}, coordinates::Axis{}}, e)};
        if constexpr (std::is_assignable_v<A, decltype(std::move(ret))>)
        {
          a = std::move(ret);
          return make_triangular_matrix<t>(std::forward<A>(a));
        }
        else return ret;
      }
    }
    else if constexpr (zero<A>)
    {
      if constexpr (diagonal_matrix<U>)
        return to_diagonal(sqrt(alpha) * diagonal_of(std::forward<U>(u)));
      else if constexpr (t == TriangleType::upper)
        return QR_decomposition(sqrt(alpha) * adjoint(std::forward<U>(u)));
      else
        return LQ_decomposition(sqrt(alpha) * std::forward<U>(u));
    }
    else if constexpr (diagonal_matrix<A> and diagonal_matrix<U>)
    {
      auto d = cholesky_factor(sum(cholesky_square(std::forward<A>(a)), alpha * cholesky_square(std::forward<U>(u))));
      if constexpr (std::is_assignable_v<A, decltype(std::move(d))>) return a = std::move(d);
      else return d;
    }
    else
    {
      auto&& an = [](A&& a) -> decltype(auto) {
        if constexpr (triangular_adapter<A>) return nested_object(std::forward<A>(a));
        else return std::forward<A>(a);
      }(std::forward<A>(a));

      auto&& aw = internal::make_writable_square_matrix<U>(std::forward<decltype(an)>(an));
      using Trait = interface::library_interface<std::decay_t<decltype(aw)>>;
      auto&& ret = Trait::template rank_update_triangular<t>(std::forward<decltype(aw)>(aw), std::forward<U>(u), alpha);
      return make_triangular_matrix<t>(std::forward<decltype(ret)>(ret));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_RANK_UPDATE_TRIANGULAR_HPP
