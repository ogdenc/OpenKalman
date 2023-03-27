/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions of Cholesky_square and Cholesky_factor for special matrices
 */

#ifndef OPENKALMAN_CHOLESKY_DECOMPOSITION_HPP
#define OPENKALMAN_CHOLESKY_DECOMPOSITION_HPP


namespace OpenKalman
{
  /**
   * \brief Take the Cholesky square of a \ref triangular_matrix.
   * \tparam A A square matrix.
   * \return AA<sup>T</sup> (if A is \ref lower_triangular_matrix) or otherwise A<sup>T</sup>A.
   */
#ifdef __cpp_concepts
  template<triangular_matrix A>
  constexpr hermitian_matrix decltype(auto)
#else
  template<typename A, std::enable_if_t<triangular_matrix<A>, int> = 0>
  constexpr decltype(auto)
#endif
  Cholesky_square(A&& a) noexcept
  {
    if constexpr (zero_matrix<A> or identity_matrix<A>)
    {
      return std::forward<A>(a);
    }
    else if constexpr (diagonal_matrix<A>)
    {
      return to_diagonal(n_ary_operation([](const auto x){
        if constexpr (complex_number<decltype(x)>) { using std::conj; return x * conj(x); }
        else return x * x;
      }, diagonal_of(std::forward<A>(a))));
    }
    else
    {
      constexpr auto triangle_type = triangle_type_of_v<A>;
      auto prod {make_dense_writable_matrix_from(OpenKalman::adjoint(a))};
      constexpr bool on_the_right = upper_triangular_matrix<A>;
      interface::LinearAlgebra<std::decay_t<A>>::template contract_in_place<on_the_right>(prod, std::forward<A>(a));
      return SelfAdjointMatrix<decltype(prod), triangle_type> {std::move(prod)};
    }
  }


  namespace detail
  {
    struct constexpr_sqrt_op
    {
      template<typename Arg>
      constexpr auto operator()(Arg arg) const noexcept { return OpenKalman::internal::constexpr_sqrt(arg); }
    };
  }


  /**
   * \brief Take the Cholesky factor of a matrix.
   * \tparam A A square matrix.
   * \tparam triangle_type Either TriangleType::upper, TriangleType::lower, or TriangleType::diagonal
   * (if A is a \ref diagonal_matrix).
   * \return T, where the argument is in the form A = TT<sup>T</sup>.
   */
#ifdef __cpp_concepts
  template<TriangleType triangle_type, hermitian_matrix A> requires (triangle_type != TriangleType::none) and
    (triangle_type != TriangleType::diagonal or diagonal_matrix<A>)
  constexpr triangular_matrix decltype(auto)
#else
  template<TriangleType triangle_type, typename A, std::enable_if_t<hermitian_matrix<A> and
    (triangle_type != TriangleType::none) and (triangle_type != TriangleType::diagonal or diagonal_matrix<A>), int> = 0>
  constexpr decltype(auto)
#endif
  Cholesky_factor(A&& a)
  {
    if constexpr (zero_matrix<A> or identity_matrix<A>)
    {
      return std::forward<A>(a);
    }
    else if constexpr (diagonal_matrix<A>)
    {
      return to_diagonal(n_ary_operation([](const auto x){ using std::sqrt; return sqrt(x); }, diagonal_of(std::forward<A>(a))));
    }
    else if constexpr (constant_matrix<A>)
    {
      // Check that Cholesky factor elements are real:
      constexpr scalar_type_of_t<A> s = constant_coefficient_v<A>;
      static_assert(s >= 0, "For a Cholesky_factor of a negative constant matrix, the constant must be positive");
      using op = internal::scalar_constant_operation<detail::constexpr_sqrt_op, constant_coefficient<A>>;

      if constexpr (triangle_type == TriangleType::diagonal)
      {
        static_assert(diagonal_matrix<A>);
        return to_diagonal(make_constant_matrix_like<A>(op{}, get_dimensions_of<0>(a), Dimensions<1>{}));
      }
      else
      {
        auto euclidean_id_a = [](const A& a) {
          constexpr auto N0 = index_dimension_of_v<A, 0>;
          constexpr auto N1 = index_dimension_of_v<A, 1>;
          if constexpr (N0 != dynamic_size) return Dimensions<N0> {};
          else if constexpr (N1 != dynamic_size) return Dimensions<N1> {};
          else return Dimensions {get_dimension_size_of<0>(a)};
        }(a);

        auto m = [](const auto& euclidean_id_a){
          constexpr Dimensions<1> D1;
          if constexpr (triangle_type == TriangleType::lower)
          {
            auto col0 = make_constant_matrix_like<A>(op{}, euclidean_id_a, D1);
            return concatenate<1>(col0, make_zero_matrix_like<A>(euclidean_id_a, euclidean_id_a - D1));
          }
          else
          {
            static_assert(triangle_type == TriangleType::upper);
            auto row0 = make_constant_matrix_like<A>(op{}, D1, euclidean_id_a);
            return concatenate<0>(row0, make_zero_matrix_like<A>(euclidean_id_a - D1, euclidean_id_a));
          }
        }(euclidean_id_a);

        auto ret = make_triangular_matrix<triangle_type>(std::move(m));
        using C0 = coefficient_types_of_t<A, 0>;
        using C1 = coefficient_types_of_t<A, 1>;
        using Cret = std::conditional_t<dynamic_index_descriptor<C0>, C1, C0>;

        if constexpr (euclidean_index_descriptor<Cret>) return ret;
        //else return make_square_root_covariance<Cret>(ret);
        else return ret; // \todo change to make_triangular_matrix
      }
    }
    else
    {
      return interface::LinearAlgebra<std::decay_t<A>>::template cholesky_factor<triangle_type>(std::forward<A>(a));
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
  template<hermitian_matrix A>
  constexpr triangular_matrix decltype(auto)
#else
  template<typename A, std::enable_if_t<hermitian_matrix<A>, int> = 0>
  constexpr decltype(auto)
#endif
  Cholesky_factor(A&& a)
  {
    constexpr auto t = hermitian_adapter_type_of_v<A>;
    constexpr auto u = diagonal_matrix<A> ? TriangleType::diagonal : t == TriangleType::none ? TriangleType::lower : t;
    return Cholesky_factor<u>(std::forward<A>(a));
  }

}


#endif //OPENKALMAN_CHOLESKY_DECOMPOSITION_HPP
