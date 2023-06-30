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

  namespace detail
  {
    template<typename U, typename A>
    constexpr decltype(auto)
    get_writable_square(A&& a)
    {
      constexpr auto dim = not dynamic_dimension<A, 0> ? index_dimension_of_v<A, 0> :
                           not dynamic_dimension<A, 1> ? index_dimension_of_v<A, 1> : index_dimension_of_v<U, 0>;
      if constexpr (writable<A>)
      {
        return std::forward<A>(a);
      }
      else if constexpr (not has_dynamic_dimensions<A> or dim == dynamic_size)
      {
        return make_dense_writable_matrix_from(std::forward<A>(a));
      }
      else
      {
        constexpr auto d = std::integral_constant<std::size_t, dim>{};
        auto ret = make_default_dense_writable_matrix_like<A>(d, d);
        ret = std::forward<A>(a);
        return ret;
      }
    }
  }


  /**
   * \brief Do a rank update on a hermitian matrix.
   * \note This may (or may not) be performed as an in-place operation if argument A is writable and hermitian.
   * \details The update is A += αUU<sup>*</sup>, returning the updated hermitian A.
   * If A is an lvalue reference, hermitian, and writable, it will be updated in place and the return value will be an
   * lvalue reference to the same, updated A. Otherwise, the function returns a new matrix.
   * \tparam A The hermitian matrix to be rank updated.
   * \tparam U The update vector or matrix.
   * \returns an updated native, writable matrix in hermitian form.
   */
#ifdef __cpp_concepts
  template<hermitian_matrix<Likelihood::maybe> A, indexible U> requires
    dimension_size_of_index_is<U, 0, index_dimension_of_v<A, 0>, Likelihood::maybe> and
    dimension_size_of_index_is<U, 0, index_dimension_of_v<A, 1>, Likelihood::maybe> and
    std::convertible_to<scalar_type_of_t<U>, const scalar_type_of_t<A>>
  inline hermitian_matrix decltype(auto)
#else
  template<typename A, typename U, std::enable_if_t<indexible<U> and hermitian_matrix<A, Likelihood::maybe> and
    dimension_size_of_index_is<U, 0, index_dimension_of<A, 0>::value, Likelihood::maybe> and
    dimension_size_of_index_is<U, 0, index_dimension_of<A, 1>::value, Likelihood::maybe> and
    std::is_convertible_v<typename scalar_type_of<U>::type, const typename scalar_type_of<A>::type>, int> = 0>
  inline decltype(auto)
#endif
  rank_update_self_adjoint(A&& a, U&& u, scalar_type_of_t<A> alpha = 1)
  {
    constexpr auto t = hermitian_adapter<A> ? hermitian_adapter_type_of_v<A> : HermitianAdapterType::lower;

    if constexpr (zero_matrix<U>)
    {
      if constexpr ((dynamic_dimension<A, 0> and dynamic_dimension<A, 0>) or dynamic_dimension<U, 0>)
        if (get_index_dimension_of<0>(a) != get_index_dimension_of<0>(u))
          throw std::invalid_argument {"In rank_update_self_adjoint, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
            ") do not match rows of u (" + std::to_string(get_index_dimension_of<0>(u)) + ")"};

      return make_hermitian_matrix<t>(std::forward<A>(a));
    }
    else if constexpr (dimension_size_of_index_is<A, 0, 1> or dimension_size_of_index_is<A, 1, 1> or dimension_size_of_index_is<U, 0, 1>)
    {
      if constexpr ((dynamic_dimension<A, 0> and dynamic_dimension<A, 0>) or dynamic_dimension<U, 0>)
        if (get_index_dimension_of<0>(a) != get_index_dimension_of<0>(u))
          throw std::invalid_argument {"In rank_update_self_adjoint, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
            ") do not match rows of u (" + std::to_string(get_index_dimension_of<0>(u)) + ")"};

      auto e = get_element(a) + alpha * get_element(contract(u, adjoint(u)));

      if constexpr (element_settable<A&&, 0>)
      {
        set_element(a, e);
        return make_hermitian_matrix<t>(std::forward<A>(a));
      }
      else
      {
        auto ret = make_dense_writable_matrix_from<A>(std::tuple{Dimensions<1>{}, Dimensions<1>{}}, e);
        if constexpr (std::is_assignable_v<A, decltype(std::move(ret))>)
        {
          a = std::move(ret);
          return make_hermitian_matrix<t>(std::forward<A>(a));
        }
        else return ret;
      }
    }
    else if constexpr (zero_matrix<A> and diagonal_matrix<U>)
    {
      if constexpr (has_dynamic_dimensions<A>) if (get_index_dimension_of<0>(a) != get_index_dimension_of<1>(a))
        throw std::invalid_argument {
          "In rank_update_self_adjoint, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
          ") do not match columns of a (" + std::to_string(get_index_dimension_of<1>(a)) + ")"};

      return alpha * Cholesky_square(std::forward<U>(u));
    }
    else if constexpr (diagonal_matrix<A> and diagonal_matrix<U>)
    {
      auto d = sum(std::forward<A>(a), alpha * Cholesky_square(std::forward<U>(u)));
      if constexpr (std::is_assignable_v<A, decltype(std::move(d))>) return a = std::move(d);
      else return d;
    }
    else if constexpr (hermitian_adapter<A>)
    {
      decltype(auto) aw = detail::get_writable_square<U>(nested_matrix(std::forward<A>(a)));
      using Trait = interface::LinearAlgebra<std::decay_t<decltype(aw)>>;
      decltype(auto) ret = Trait::template rank_update_self_adjoint<t>(std::forward<decltype(aw)>(aw), std::forward<U>(u), alpha);
      return make_hermitian_matrix<t>(std::forward<decltype(ret)>(ret));
    }
    else // hermitian_matrix but not hermitian_adapter
    {
      decltype(auto) aw = detail::get_writable_square<U>(std::forward<A>(a));
      using Trait = interface::LinearAlgebra<std::decay_t<decltype(aw)>>;
      decltype(auto) ret = Trait::template rank_update_self_adjoint<t>(std::forward<decltype(aw)>(aw), std::forward<U>(u), alpha);
      return make_hermitian_matrix<t>(std::forward<decltype(ret)>(ret));
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
  template<triangular_matrix<TriangleType::any, Likelihood::maybe> A, indexible U> requires
    dimension_size_of_index_is<U, 0, index_dimension_of_v<A, 0>, Likelihood::maybe> and
    dimension_size_of_index_is<U, 0, index_dimension_of_v<A, 1>, Likelihood::maybe> and
    std::convertible_to<scalar_type_of_t<U>, const scalar_type_of_t<A>>
  inline triangular_matrix<triangle_type_of_v<A> == TriangleType::upper ? TriangleType::upper : TriangleType::lower> decltype(auto)
# else
  template<typename A, typename U, std::enable_if_t<triangular_matrix<A, TriangleType::any, Likelihood::maybe> and indexible<U> and
    dimension_size_of_index_is<U, 0, index_dimension_of<A, 0>::value, Likelihood::maybe> and
    dimension_size_of_index_is<U, 0, index_dimension_of<A, 1>::value, Likelihood::maybe> and
    std::is_convertible_v<scalar_type_of_t<U>, const scalar_type_of_t<A>>, int> = 0>
  inline decltype(auto)
# endif
  rank_update_triangular(A&& a, U&& u, scalar_type_of_t<A> alpha = 1)
  {
    using std::sqrt;

    constexpr auto t = triangle_type_of_v<A> == TriangleType::upper ? TriangleType::upper : TriangleType::lower;

    if constexpr (zero_matrix<U>)
    {
      if constexpr (dynamic_rows<A> or dynamic_rows<U>) if (get_index_dimension_of<0>(a) != get_index_dimension_of<0>(u))
        throw std::invalid_argument {"In rank_update_triangular, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
          ") do not match rows of u (" + std::to_string(get_index_dimension_of<0>(u)) + ")"};

      return make_triangular_matrix<t>(std::forward<A>(a));
    }
    else if constexpr (dimension_size_of_index_is<A, 0, 1> or dimension_size_of_index_is<A, 1, 1> or dimension_size_of_index_is<U, 0, 1>)
    {
      if constexpr (dynamic_rows<A> or dynamic_rows<U>) if (get_index_dimension_of<0>(a) != get_index_dimension_of<0>(u))
        throw std::invalid_argument {"In rank_update_triangular, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
          ") do not match rows of u (" + std::to_string(get_index_dimension_of<0>(u)) + ")"};

      // From here on, A is known to be a 1-by-1 matrix.
      auto e = [](const auto& a, auto&& uterm) {
          using std::conj;
          if constexpr (complex_number<scalar_type_of<A>>) return sqrt(get_element(a) * conj(get_element(a)) + uterm);
          else return sqrt(get_element(a) * get_element(a) + uterm);
      }(a, alpha * get_element(contract(u, adjoint(u))));

      if constexpr (element_settable<A&&, 0>)
      {
        set_element(a, e);
        return make_triangular_matrix<t>(std::forward<A>(a));
      }
      else
      {
        auto ret = make_dense_writable_matrix_from<A>(std::tuple{Dimensions<1>{}, Dimensions<1>{}}, e);
        if constexpr (std::is_assignable_v<A, decltype(std::move(ret))>)
        {
          a = std::move(ret);
          return make_triangular_matrix<t>(std::forward<A>(a));
        }
        else return ret;
      }
    }
    else if constexpr (zero_matrix<A>)
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
      auto d = Cholesky_factor(sum(Cholesky_square(std::forward<A>(a)), alpha * Cholesky_square(std::forward<U>(u))));
      if constexpr (std::is_assignable_v<A, decltype(std::move(d))>) return a = std::move(d);
      else return d;
    }
    else
    {
      decltype(auto) an = [](A&& a) -> decltype(auto) {
        if constexpr (triangular_adapter<A>) return nested_matrix(std::forward<A>(a));
        else return std::forward<A>(a);
      }(std::forward<A>(a));

      decltype(auto) aw = detail::get_writable_square<U>(std::forward<decltype(an)>(an));
      using Trait = interface::LinearAlgebra<std::decay_t<decltype(aw)>>;
      decltype(auto) ret = Trait::template rank_update_triangular<t>(std::forward<decltype(aw)>(aw), std::forward<U>(u), alpha);
      return make_triangular_matrix<t>(std::forward<decltype(ret)>(ret));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_RANK_UPDATE_HPP
