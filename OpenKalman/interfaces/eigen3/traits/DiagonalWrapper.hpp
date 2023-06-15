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
 * \brief Type traits as applied to Eigen::DiagonalWrapper.
 */

#ifndef OPENKALMAN_EIGEN3_TRAITS_DIAGONALWRAPPER_HPP
#define OPENKALMAN_EIGEN3_TRAITS_DIAGONALWRAPPER_HPP

#include <type_traits>


namespace OpenKalman
{
  namespace interface
  {
    template<typename DiagVectorType>
    struct IndexTraits<Eigen::DiagonalWrapper<DiagVectorType>>
    {
      static constexpr std::size_t max_indices = 2;

      template<std::size_t N, typename Arg>
      static constexpr auto get_index_descriptor(const Arg& arg)
      {
        if constexpr (has_dynamic_dimensions<DiagVectorType>) return static_cast<std::size_t>(arg.rows());
        else return Dimensions<index_dimension_of_v<DiagVectorType, 0> * index_dimension_of_v<DiagVectorType, 1>>{};
      }

      template<std::size_t N>
      static constexpr std::size_t dimension = has_dynamic_dimensions<DiagVectorType> ? dynamic_size :
        index_dimension_of_v<DiagVectorType, 0> * index_dimension_of_v<DiagVectorType, 1>;

      template<std::size_t N, typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        if constexpr (dimension<N> == dynamic_size) return static_cast<std::size_t>(arg.rows());
        else return dimension<N>;
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one = one_by_one_matrix<DiagVectorType, b>;

      template<Likelihood b>
      static constexpr bool is_square = true;
    };


    template<typename DiagVectorType>
    struct Dependencies<Eigen::DiagonalWrapper<DiagVectorType>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename DiagVectorType::Nested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        decltype(auto) d = std::forward<Arg>(arg).diagonal();
        using D = decltype(d);
        using NCD = std::conditional_t<
          std::is_const_v<std::remove_reference_t<Arg>> or std::is_const_v<DiagVectorType>,
          D, std::conditional_t<std::is_lvalue_reference_v<D>, std::decay_t<D>&, std::decay_t<D>>>;
        return const_cast<NCD>(std::forward<decltype(d)>(d));
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        auto d {make_self_contained(std::forward<Arg>(arg).diagonal())};
        return DiagonalMatrix<decltype(d)> {d};
      }
    };


    template<typename DiagVectorType>
    struct SingleConstant<Eigen::DiagonalWrapper<DiagVectorType>>
    {
      const Eigen::DiagonalWrapper<DiagVectorType>& xpr;

      constexpr auto get_constant_diagonal()
      {
        return constant_coefficient {xpr.diagonal()};
      }
    };


    template<typename DiagVectorType>
    struct TriangularTraits<Eigen::DiagonalWrapper<DiagVectorType>>
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = true;

      static constexpr bool is_triangular_adapter = false;

      static constexpr bool is_diagonal_adapter = true;
    };


    template<typename DiagVectorType>
    struct Conversions<Eigen::DiagonalWrapper<DiagVectorType>>
    {
      template<typename Arg>
      static constexpr decltype(auto) to_diagonal(Arg&& arg)
      {
        // In this case, arg will be a one-by-one matrix.
        if constexpr (has_dynamic_dimensions<DiagVectorType>)
          if (get_index_dimension_of<0>(arg) + get_index_dimension_of<1>(arg) != 1) throw std::logic_error {
            "Argument of to_diagonal must have 1 element; instead it has " + std::to_string(get_index_dimension_of<1>(arg))};

        return make_self_contained<Arg>(std::forward<Arg>(arg).diagonal());
      }


      template<typename Arg>
      static constexpr decltype(auto) diagonal_of(Arg&& arg)
      {
        using Scalar = scalar_type_of_t<Arg>;
        decltype(auto) diag {nested_matrix(std::forward<Arg>(arg))}; //< must be nested_matrix(...) rather than .diagonal() because of const_cast
        using Diag = decltype(diag);
        using EigenTraits = Eigen::internal::traits<std::decay_t<Diag>>;
        constexpr Eigen::Index rows = EigenTraits::RowsAtCompileTime;
        constexpr Eigen::Index cols = EigenTraits::ColsAtCompileTime;

        if constexpr (cols == 1 or cols == 0)
        {
          return std::forward<Diag>(diag);
        }
        else if constexpr (rows == 1 or rows == 0)
        {
          return transpose(std::forward<Diag>(diag));
        }
        else if constexpr (rows == Eigen::Dynamic or cols == Eigen::Dynamic)
        {
          auto d {make_dense_writable_matrix_from(std::forward<Diag>(diag))};
          using M = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
          return M {M::Map(make_dense_writable_matrix_from(std::forward<Diag>(diag)).data(),
            get_index_dimension_of<0>(diag) * get_index_dimension_of<1>(diag))};
        }
        else // rows > 1 and cols > 1
        {
          using M = Eigen::Matrix<Scalar, rows * cols, 1>;
          return M {M::Map(make_dense_writable_matrix_from(std::forward<Diag>(diag)).data())};
        }
      }
    };

  } // namespace interface


  /**
   * \internal
   * \brief Matrix traits for Eigen::DiagonalWrapper.
   */
  template<typename V>
  struct MatrixTraits<Eigen::DiagonalWrapper<V>>
    : MatrixTraits<Eigen::Matrix<typename Eigen::internal::traits<std::decay_t<V>>::Scalar,
        V::SizeAtCompileTime, V::SizeAtCompileTime>> {};

} // namespace OpenKalman

#endif //OPENKALMAN_EIGEN3_TRAITS_DIAGONALWRAPPER_HPP
