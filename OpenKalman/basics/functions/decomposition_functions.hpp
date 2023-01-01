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
 * \brief Overloaded general linear-algebra functions.
 */

#ifndef OPENKALMAN_DECOMPOSITION_FUNCTIONS_HPP
#define OPENKALMAN_DECOMPOSITION_FUNCTIONS_HPP

namespace OpenKalman
{
  using namespace interface;

  /**
   * \brief Perform an LQ decomposition of matrix A=[L,0]Q, L is a lower-triangular matrix, and Q is orthogonal.
   * \tparam A The matrix to be decomposed
   * \returns L as a \ref lower_triangular_matrix
   */
#ifdef __cpp_concepts
  template<indexible A> requires (not euclidean_transformed<A>)
  constexpr lower_triangular_matrix<Likelihood::maybe> auto
#else
  template<typename A, std::enable_if_t<indexible<A> and (not euclidean_transformed<A>), int> = 0>
  constexpr auto
#endif
  LQ_decomposition(A&& a)
  {
    if constexpr (lower_triangular_matrix<A>)
    {
      return std::forward<A>(a);
    }
    else if constexpr (zero_matrix<A>)
    {
      auto dim = get_dimensions_of<0>(a);
      return make_zero_matrix_like<A>(dim, dim);
    }
    else if constexpr (constant_matrix<A>)
    {
      using Scalar = scalar_type_of_t<A>;
      constexpr auto constant = constant_coefficient_v<A>;

      const Scalar elem = constant * [](A&& a){
        if constexpr (dynamic_dimension<A, 1>) return square_root(static_cast<Scalar>(get_index_dimension_of<1>(a)));
        else return internal::constexpr_sqrt(static_cast<Scalar>(index_dimension_of_v<A, 1>));
      }(std::forward<A>(a));

      if constexpr (dynamic_dimension<A, 0>)
      {
        auto dim = Dimensions {get_index_dimension_of<0>(a)};
        auto col1 = elem * make_constant_matrix_like<A, 1>(dim, Dimensions<1>{});

        auto m {make_default_dense_writable_matrix_like<A>(dim, dim)};

        if (get_dimension_size_of(dim) == 1) m = std::move(col1);
        else m = concatenate<1>(std::move(col1), make_zero_matrix_like<A>(dim, dim - Dimensions<1>{}));

        auto ret = make_triangular_matrix<TriangleType::lower>(std::move(m));

        // \todo Fix this:
        if constexpr (euclidean_index_descriptor<coefficient_types_of_t<A, 0>>) return ret;
        else return SquareRootCovariance {std::move(ret), get_dimensions_of<0>(a)};
      }
      else
      {
        auto ret = make_triangular_matrix<TriangleType::lower>([](Scalar elem){
          constexpr auto dim = index_dimension_of_v<A, 0>;
          auto col1 = elem * make_constant_matrix_like<A, 1>(Dimensions<dim>{}, Dimensions<1>{});
          if constexpr (dim == 1) return col1;
          else return concatenate<1>(std::move(col1), make_zero_matrix_like<A>(Dimensions<dim>{}, Dimensions<dim - 1>{}));
        }(elem));

        // \todo Fix this:
        using C = coefficient_types_of_t<A, 0>;
        if constexpr (euclidean_index_descriptor<C>) return ret;
        else return SquareRootCovariance {std::move(ret), C{}};
      }
    }
    else
    {
      auto ret {interface::LinearAlgebra<std::decay_t<A>>::LQ_decomposition(std::forward<A>(a))};
      static_assert(lower_triangular_matrix<decltype(ret)>,
        "Interface implementation error: interface::LinearAlgebra<T>::LQ_decomposition must return a lower_triangular_matrix.");

      // \todo Fix this:
      if constexpr (euclidean_index_descriptor<coefficient_types_of_t<A, 0>>) return ret;
      else return SquareRootCovariance {std::move(ret), get_dimensions_of<0>(a)};
    }
  }


  /**
   * \brief Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
   * \tparam A The matrix to be decomposed
   * \returns U as an \ref upper_triangular_matrix
   */
#ifdef __cpp_concepts
  template<indexible A>
  constexpr upper_triangular_matrix<Likelihood::maybe> auto
#else
  template<typename A, std::enable_if_t<indexible<A>, int> = 0>
  constexpr auto
#endif
  QR_decomposition(A&& a)
  {
    if constexpr (upper_triangular_matrix<A>)
    {
      return std::forward<A>(a);
    }
    else if constexpr (zero_matrix<A>)
    {
      auto dim = get_dimensions_of<1>(a);
      return make_zero_matrix_like<A>(dim, dim);
    }
    else if constexpr (constant_matrix<A>)
    {
      using Scalar = scalar_type_of_t<A>;
      constexpr auto constant = constant_coefficient_v<A>;

      const Scalar elem = constant * [](A&& a){
        if constexpr (dynamic_dimension<A, 0>) return square_root(static_cast<Scalar>(get_index_dimension_of<0>(a)));
        else return internal::constexpr_sqrt(static_cast<Scalar>(index_dimension_of_v<A, 0>));
      }(std::forward<A>(a));

      if constexpr (dynamic_dimension<A, 1>)
      {
        auto dim = Dimensions {get_index_dimension_of<1>(a)};
        auto row1 = elem * make_constant_matrix_like<A, 1>(Dimensions<1>{}, dim);

        auto m = make_default_dense_writable_matrix_like<A>(dim, dim);

        if (get_dimension_size_of(dim) == 1) m = std::move(row1);
        else m = concatenate<0>(std::move(row1), make_zero_matrix_like<A>(dim - Dimensions<1>{}, dim));

        auto ret = make_triangular_matrix<TriangleType::upper>(std::move(m));

        // \todo Fix this:
        if constexpr (euclidean_index_descriptor<coefficient_types_of_t<A, 1>>) return ret;
        else return SquareRootCovariance {std::move(ret), get_dimensions_of<1>(a)};
      }
      else
      {
        auto ret = make_triangular_matrix<TriangleType::upper>([](Scalar elem){
          constexpr auto dim = index_dimension_of_v<A, 1>;
          auto row1 = elem * make_constant_matrix_like<A, 1>(Dimensions<1>{}, Dimensions<dim>{});
          if constexpr (dim == 1) return row1;
          else return concatenate<0>(std::move(row1), make_zero_matrix_like<A>(Dimensions<dim - 1>{}, Dimensions<dim>{}));
        }(elem));

        // \todo Fix this:
        using C = coefficient_types_of_t<A, 1>;
        if constexpr (euclidean_index_descriptor<C>) return ret;
        else return SquareRootCovariance {std::move(ret), C{}};
      }
    }
    else
    {
      auto ret {interface::LinearAlgebra<std::decay_t<A>>::QR_decomposition(std::forward<A>(a))};
      static_assert(upper_triangular_matrix<decltype(ret)>,
        "Interface implementation error: interface::LinearAlgebra<T>::QR_decomposition must return an upper_triangular_matrix.");

      // \todo Fix this:
      if constexpr (euclidean_index_descriptor<coefficient_types_of_t<A, 1>>) return ret;
      else return SquareRootCovariance {std::move(ret), get_dimensions_of<1>(a)};
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_DECOMPOSITION_FUNCTIONS_HPP
