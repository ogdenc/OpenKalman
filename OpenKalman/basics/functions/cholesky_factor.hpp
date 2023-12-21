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
 * \brief Definition of cholesky_factor.
 */

#ifndef OPENKALMAN_CHOLESKY_FACTOR_HPP
#define OPENKALMAN_CHOLESKY_FACTOR_HPP


namespace OpenKalman
{
  /**
   * \brief Take the Cholesky factor of a matrix.
   * \tparam A A square matrix.
   * \tparam triangle_type Either TriangleType::upper, TriangleType::lower, or TriangleType::diagonal
   * (if A is a \ref diagonal_matrix).
   * \return T, where the argument is in the form A = TT<sup>T</sup>.
   */
#ifdef __cpp_concepts
  template<TriangleType triangle_type, hermitian_matrix A> requires
    (triangle_type != TriangleType::diagonal or diagonal_matrix<A>)
  constexpr triangular_matrix<triangle_type> decltype(auto)
#else
  template<TriangleType triangle_type, typename A, std::enable_if_t<hermitian_matrix<A> and
    (triangle_type != TriangleType::diagonal or diagonal_matrix<A>), int> = 0>
  constexpr decltype(auto)
#endif
  cholesky_factor(A&& a)
  {
    if constexpr (zero<A> or identity_matrix<A>)
    {
      return std::forward<A>(a);
    }
    else if constexpr (constant_diagonal_matrix<A>)
    {
      auto sq = internal::constexpr_sqrt(constant_diagonal_coefficient{a});
      return to_diagonal(make_constant<A>(sq, get_vector_space_descriptor<0>(a), Dimensions<1>{}));
    }
    else if constexpr (constant_matrix<A>)
    {
      auto m = [](const auto& a){
        auto euclidean_id_a = [](const A& a) {
          constexpr auto N0 = index_dimension_of_v<A, 0>;
          constexpr auto N1 = index_dimension_of_v<A, 1>;
          if constexpr (N0 != dynamic_size) return Dimensions<N0> {};
          else if constexpr (N1 != dynamic_size) return Dimensions<N1> {};
          else return Dimensions {get_dimension_size_of<0>(a)};
        }(a);

        auto sq = internal::constexpr_sqrt(constant_coefficient{a});

        constexpr Dimensions<1> D1;
        if constexpr (triangle_type == TriangleType::lower)
        {
          auto col0 = make_constant<A>(sq, euclidean_id_a, D1);
          return concatenate<1>(col0, make_zero<A>(euclidean_id_a, euclidean_id_a - D1));
        }
        else
        {
          static_assert(triangle_type == TriangleType::upper);
          auto row0 = make_constant<A>(sq, D1, euclidean_id_a);
          return concatenate<0>(row0, make_zero<A>(euclidean_id_a - D1, euclidean_id_a));
        }
      }(a);

      auto ret {make_triangular_matrix<triangle_type>(std::move(m))};
      using C0 = vector_space_descriptor_of_t<A, 0>;
      using C1 = vector_space_descriptor_of_t<A, 1>;
      using Cret = std::conditional_t<dynamic_vector_space_descriptor<C0>, C1, C0>;

      if constexpr (euclidean_vector_space_descriptor<Cret>) return ret;
      //else return make_square_root_covariance<Cret>(ret);
      else return ret; // \todo change to make_triangular_matrix
    }
    else if constexpr (diagonal_matrix<A>)
    {
      // \todo Add facility to potentially use native library operators such as a square-root operator.
      return to_diagonal(n_ary_operation([](const auto x){ using std::sqrt; return sqrt(x); }, diagonal_of(std::forward<A>(a))));
    }
    else
    {
      return interface::library_interface<std::decay_t<A>>::template cholesky_factor<triangle_type>(std::forward<A>(a));
    }
  }


 /**
  * \overload
  * \details This overload does not require specifying the TriangleType, which is either
  * # TriangleType::diagonal if A is diagonal;
  * # the hermitian adapter triangle type of A, if it exists; or
  * # TriangleType::lower, by default.
  */
#ifdef __cpp_concepts
  template<hermitian_matrix A> requires hermitian_adapter<A> or diagonal_matrix<A>
  constexpr triangular_matrix decltype(auto)
#else
  template<typename A, std::enable_if_t<hermitian_adapter<A> or diagonal_matrix<A>, int> = 0>
  constexpr decltype(auto)
#endif
  cholesky_factor(A&& a)
  {
    constexpr auto u = diagonal_matrix<A> ? TriangleType::diagonal :
      hermitian_adapter<A, HermitianAdapterType::upper> ? TriangleType::upper : TriangleType::lower;
    return cholesky_factor<u>(std::forward<A>(a));
  }

} // namespace OpenKalman


#endif //OPENKALMAN_CHOLESKY_FACTOR_HPP
