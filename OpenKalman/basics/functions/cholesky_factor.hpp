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
   * \tparam A A hermitian matrix.
   * \tparam triangle_type Either TriangleType::upper, TriangleType::lower, or TriangleType::diagonal
   * (only if A is a \ref diagonal_matrix).
   * \return T, where the argument is in the form A = TT<sup>T</sup>.
   */
#ifdef __cpp_concepts
  template<TriangleType triangle_type, hermitian_matrix<Qualification::depends_on_dynamic_shape> A> requires
    (triangle_type != TriangleType::diagonal or diagonal_matrix<A>)
  constexpr triangular_matrix<triangle_type> decltype(auto)
#else
  template<TriangleType triangle_type, typename A, std::enable_if_t<hermitian_matrix<A, Qualification::depends_on_dynamic_shape> and
    (triangle_type != TriangleType::diagonal or diagonal_matrix<A>), int> = 0>
  constexpr decltype(auto)
#endif
  cholesky_factor(A&& a)
  {
    if constexpr (not square_shaped<A>)
      if (not is_square_shaped(a)) throw std::invalid_argument {"Argument to cholesky_factor must be a square matrix"};

    if constexpr (zero<A> or identity_matrix<A>)
    {
      return std::forward<A>(a);
    }
    else if constexpr (constant_diagonal_matrix<A>)
    {
      auto sq = internal::constexpr_sqrt(constant_diagonal_coefficient{a});
      return to_diagonal(make_constant<A>(sq, get_vector_space_descriptor<0>(a)));
    }
    else if constexpr (constant_matrix<A>)
    {
      auto m = [](const auto& a){
        auto sq = internal::constexpr_sqrt(constant_coefficient{a});
        auto v = *is_square_shaped(a);
        auto dim = get_dimension_size_of(v);

        if constexpr (triangle_type == TriangleType::lower)
        {
          auto col0 = make_constant<A>(sq, dim, Dimensions<1>{});
          return make_vector_space_adapter(concatenate<1>(col0, make_zero<A>(dim, dim - Dimensions<1>{})), v, v);
        }
        else
        {
          static_assert(triangle_type == TriangleType::upper);
          auto row0 = make_constant<A>(sq, Dimensions<1>{}, dim);
          return make_vector_space_adapter(concatenate<0>(row0, make_zero<A>(dim - Dimensions<1>{}, dim)), v, v);
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
  template<hermitian_matrix<Qualification::depends_on_dynamic_shape> A>
  constexpr triangular_matrix decltype(auto)
#else
  template<typename A, std::enable_if_t<hermitian_matrix<A, Qualification::depends_on_dynamic_shape>, int> = 0>
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
