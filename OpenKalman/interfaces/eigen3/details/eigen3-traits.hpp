/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Type traits as applied to native Eigen3 types.
 */

#ifndef OPENKALMAN_EIGEN3_TRAITS_HPP
#define OPENKALMAN_EIGEN3_TRAITS_HPP

#include <type_traits>


namespace OpenKalman
{
  using namespace OpenKalman::internal;
  using namespace OpenKalman::Eigen3;
  using namespace OpenKalman::Eigen3::internal;


  namespace interface
  {

  // ---------------------- //
  //  native_eigen_general  //
  // ---------------------- //

#ifdef __cpp_concepts
    template<Eigen3::native_eigen_general T>
    struct RowExtentOf<T>
#else
    template<typename T>
    struct RowExtentOf<T, std::enable_if_t<Eigen3::native_eigen_matrix<T>>>
#endif
      : std::integral_constant<std::size_t,
        Eigen::internal::traits<std::decay_t<T>>::RowsAtCompileTime == Eigen::Dynamic ?
        dynamic_extent : static_cast<std::size_t>(Eigen::internal::traits<std::decay_t<T>>::RowsAtCompileTime)>
    {
      template<typename Arg>
      static constexpr std::size_t rows_at_runtime(Arg&& arg)
      {
        static_assert(std::is_same_v<std::decay_t<Arg>, std::decay_t<T>>);
        if constexpr (dynamic_rows<Arg>)
        {
          if constexpr (square_matrix<Arg> and not dynamic_columns<Arg>)
            return column_extent_of_v<Arg>;
          else
            return arg.rows();
        }
        else
        {
          return RowExtentOf::value;
        }
      }
    };


#ifdef __cpp_concepts
    template<Eigen3::native_eigen_general T>
    struct ColumnExtentOf<T>
#else
    template<typename T>
    struct ColumnExtentOf<T, std::enable_if_t<Eigen3::native_eigen_matrix<T>>>
#endif
      : std::integral_constant<std::size_t,
        Eigen::internal::traits<std::decay_t<T>>::ColsAtCompileTime == Eigen::Dynamic ?
        dynamic_extent : static_cast<std::size_t>(Eigen::internal::traits<std::decay_t<T>>::ColsAtCompileTime)>
    {
        template<typename Arg>
        static constexpr std::size_t columns_at_runtime(Arg&& arg)
        {
          static_assert(std::is_same_v<std::decay_t<Arg>, std::decay_t<T>>);
          if constexpr (dynamic_columns<Arg>)
          {
            if constexpr (square_matrix<Arg> and not dynamic_rows<Arg>)
              return row_extent_of_v<Arg>;
            else
              return arg.cols();
          }
          else
          {
            return ColumnExtentOf::value;
          }
        }
    };


#ifdef __cpp_concepts
    template<Eigen3::native_eigen_general T>
    struct ScalarTypeOf<T>
#else
    template<typename T>
    struct ScalarTypeOf<T, std::enable_if_t<Eigen3::native_eigen_matrix<T>>>
#endif
    {
      using type = typename Eigen::internal::traits<std::decay_t<T>>::Scalar;
    };


#ifdef __cpp_concepts
    template<Eigen3::native_eigen_general T, std::size_t row_extent, std::size_t column_extent, typename scalar_type>
    struct EquivalentDenseWritableMatrix<T, row_extent, column_extent, scalar_type>
#else
    template<typename T, std::size_t row_extent, std::size_t column_extent, typename scalar_type>
    struct EquivalentDenseWritableMatrix<T, row_extent, column_extent, scalar_type, std::enable_if_t<
      Eigen3::native_eigen_general<T>>>
#endif
    {
    private:

      template<typename S, typename Scalar, auto...Args>
      using dense_type = std::conditional_t<native_eigen_array<S>,
        Eigen::Array<Scalar, Args...>, Eigen::Matrix<Scalar, Args...>>;

    public:

      using type = dense_type<T, scalar_type,
        row_extent == dynamic_extent ? Eigen::Dynamic : (Eigen::Index) row_extent,
        column_extent == dynamic_extent ? Eigen::Dynamic : (Eigen::Index) column_extent,
        ((Eigen::internal::traits<std::decay_t<T>>::Flags & Eigen::RowMajorBit) == Eigen::RowMajorBit ?
          Eigen::RowMajor : Eigen::ColMajor) | Eigen::AutoAlign>;


#ifdef __cpp_concepts
      template<std::convertible_to<std::size_t>...extents> requires
        (sizeof...(extents) == (row_extent == dynamic_extent ? 1 : 0) + (column_extent == dynamic_extent ? 1 : 0))
#else
      template<std::size_t rows = row_extent, typename...extents, std::enable_if_t<sizeof...(extents) ==
        (rows == dynamic_extent ? 1 : 0) + (column_extent == dynamic_extent ? 1 : 0) and
        (std::is_convertible_v<extents, std::size_t> and ...), int> = 0>
#endif
      static type make_default(extents...e)
      {
        if constexpr (row_extent == dynamic_extent)
        {
          if constexpr (column_extent == dynamic_extent)
            return type(e...);
          else
            return type(e..., column_extent);
        }
        else
        {
          if constexpr (column_extent == dynamic_extent)
            return type(row_extent, e...);
          else
            return type(); // fixed shape
        }
      }


#ifdef __cpp_concepts
      template<typename Arg> requires
        (row_extent_of_v<Arg> == dynamic_extent or row_extent_of_v<Arg> == row_extent) and
        (column_extent_of_v<Arg> == dynamic_extent or column_extent_of_v<Arg> == column_extent) and
        std::convertible_to<scalar_type_of_t<Arg>, scalar_type>
 #else
      template<typename Arg, std::enable_if_t<
        (row_extent_of<Arg>::value == dynamic_extent or row_extent_of<Arg>::value == row_extent) and
        (column_extent_of<Arg>::value == dynamic_extent or column_extent_of<Arg>::value == column_extent) and
        std::is_convertible_v<typename scalar_type_of<Arg>::type, scalar_type>, int> = 0>
 #endif
      static decltype(auto) convert(Arg&& arg)
      {
        if constexpr (eigen_DiagonalWrapper<Arg>)
        {
          using Scalar = scalar_type_of_t<Arg>;
          auto& diag = std::forward<Arg>(arg).diagonal();
          using Diag = std::decay_t<decltype(diag)>;
          constexpr Eigen::Index rows = Eigen::internal::traits<Diag>::RowsAtCompileTime;
          constexpr Eigen::Index cols = Eigen::internal::traits<Diag>::ColsAtCompileTime;

          if constexpr (cols == 1 or cols == 0)
          {
            return type(Eigen::DiagonalWrapper {diag});
          }
          else if constexpr (rows == 1 or rows == 0)
          {
            return type(Eigen::DiagonalWrapper {diag.transpose()});
          }
          else if constexpr (rows == Eigen::Dynamic or cols == Eigen::Dynamic)
          {
            using M = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
            if constexpr (std::is_base_of_v<Eigen::PlainObjectBase<Diag>, Diag>)
            {
              return type(Eigen3::DiagonalMatrix {M::Map(diag.data(), (Eigen::Index) (diag.rows() * diag.cols()))});
            }
            else
            {
              auto d = make_native_matrix(diag);
              return type(Eigen3::DiagonalMatrix {M::Map(d.data(), (Eigen::Index) (d.rows() * d.cols()))});
            }

          }
          else // rows > 1 and cols > 1
          {
            using M = Eigen::Matrix<Scalar, rows * cols, 1>;
            if constexpr (std::is_base_of_v<Eigen::PlainObjectBase<Diag>, Diag>)
            {
              return type(Eigen3::DiagonalMatrix {M::Map(diag.data())});
            }
            else
            {
              auto d = make_native_matrix(diag);
              return type(Eigen3::DiagonalMatrix {M::Map(d.data())});
            }
          }

          // \todo After universalizing diagonal_of and to_diagonal, replace the above with this:
          // Because Arg might not be a column vector:
          //return type {to_diagonal(diagonal_of(std::forward<Arg>(arg)))};
        }
        else
        {
          return type(std::forward<Arg>(arg));
        }
      }

    };


    // ------- //
    //  Array  //
    // ------- //

    template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
    struct Dependencies<Eigen::Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    {
      static constexpr bool has_runtime_parameters = true;
      using type = std::tuple<>;
    };


    // -------------- //
    //  ArrayWrapper  //
    // -------------- //

    template<typename XprType>
    struct Dependencies<Eigen::ArrayWrapper<XprType>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename Eigen::ArrayWrapper<XprType>::NestedExpressionType>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        using N = Eigen::ArrayWrapper<equivalent_self_contained_t<XprType>>;
        if constexpr (not std::is_lvalue_reference_v<typename N::NestedExpressionType>)
          return N {make_self_contained(arg.nestedExpression())};
        else
          return make_native_matrix(std::forward<Arg>(arg));
      }
    };


  } // namespace interface


#ifdef __cpp_concepts
  template<constant_matrix XprType>
  struct constant_coefficient<Eigen::ArrayWrapper<XprType>>
#else
  template<typename XprType>
  struct constant_coefficient<Eigen::ArrayWrapper<XprType>, std::enable_if_t<constant_matrix<XprType>>>
#endif
    : constant_coefficient<std::decay_t<XprType>> {};


#ifdef __cpp_concepts
  template<constant_diagonal_matrix XprType>
  struct constant_diagonal_coefficient<Eigen::ArrayWrapper<XprType>>
#else
  template<typename XprType>
  struct constant_diagonal_coefficient<Eigen::ArrayWrapper<XprType>, std::enable_if_t<
    constant_diagonal_matrix<XprType>>>
#endif
    : constant_diagonal_coefficient<std::decay_t<XprType>> {};


  template<typename XprType>
  struct is_diagonal_matrix<Eigen::ArrayWrapper<XprType>>
    : std::bool_constant<diagonal_matrix<XprType>> {};


  template<typename XprType>
  struct is_lower_self_adjoint_matrix<Eigen::ArrayWrapper<XprType>>
    : std::bool_constant<lower_self_adjoint_matrix<XprType>> {};


  template<typename XprType>
  struct is_upper_self_adjoint_matrix<Eigen::ArrayWrapper<XprType>>
    : std::bool_constant<upper_self_adjoint_matrix<XprType>> {};


  template<typename XprType>
  struct is_lower_triangular_matrix<Eigen::ArrayWrapper<XprType>>
    : std::bool_constant<lower_triangular_matrix<XprType>> {};


  template<typename XprType>
  struct is_upper_triangular_matrix<Eigen::ArrayWrapper<XprType>>
    : std::bool_constant<upper_triangular_matrix<XprType>> {};


  // ------- //
  //  Block  //
  // ------- //

  namespace interface
  {
    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    struct Dependencies<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
    {
      static constexpr bool has_runtime_parameters = true;
      using type = std::tuple<typename Eigen::internal::ref_selector<XprType>::non_const_type>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      // Eigen::Block should always be converted to Matrix

    };
  }


  /// A block taken from a constant matrix is constant.
#ifdef __cpp_concepts
  template<constant_matrix XprType, int BlockRows, int BlockCols, bool InnerPanel>
  struct constant_coefficient<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
#else
  template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
  struct constant_coefficient<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>,
    std::enable_if_t<constant_matrix<XprType>>>
#endif
    : constant_coefficient<std::decay_t<XprType>> {};


  /// A block taken from a constant matrix is constant-diagonal if it is square and either zero or one-by-one.
#ifdef __cpp_concepts
  template<constant_matrix XprType, int BlockRows, int BlockCols, bool InnerPanel>
  requires (BlockRows == BlockCols) and (constant_coefficient_v<XprType> == 0 or BlockRows == 1 or BlockCols == 1)
  struct constant_diagonal_coefficient<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
#else
  template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
  struct constant_diagonal_coefficient<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>, std::enable_if_t<
    constant_matrix<XprType> and (BlockRows == BlockCols) and
    (zero_matrix<XprType> or BlockRows == 1 or BlockCols == 1)>>
#endif
    : constant_coefficient<std::decay_t<XprType>> {};


  // --------------- //
  //  CwiseBinaryOp  //
  // --------------- //

  namespace interface
  {
    template<typename BinaryOp, typename LhsType, typename RhsType>
    struct Dependencies<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
    {
    private:

      using T = Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>;
      using N = Eigen::CwiseBinaryOp<BinaryOp, equivalent_self_contained_t<LhsType>,
        equivalent_self_contained_t<RhsType>>;

    public:

      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename T::LhsNested, typename T::RhsNested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i < 2);
        if constexpr (i == 0)
          return std::forward<Arg>(arg).lhs();
        else
          return std::forward<Arg>(arg).rhs();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        // Do a partial evaluation as long as at least one argument is already self-contained.
        if constexpr ((self_contained<LhsType> or self_contained<RhsType>) and
          not std::is_lvalue_reference_v<typename N::LhsNested> and
          not std::is_lvalue_reference_v<typename N::RhsNested>)
        {
          return N {make_self_contained(arg.lhs()), make_self_contained(arg.rhs()), arg.functor()};
        }
        else
        {
          return make_native_matrix(std::forward<Arg>(arg));
        }
      }
    };
  }


  // --- constant_coefficient --- //


  /// The sum of two constant matrices is also constant.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg1> + constant_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_sum_op<Scalar1, Scalar2>::result_type> {};


  /// The sum of two constant-diagonal matrices is zero if the matrices cancel out.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
  requires (not constant_matrix<Arg1>) and (not constant_matrix<Arg2>) and
    (constant_diagonal_coefficient_v<Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>,
      Arg1, Arg2>> == 0)
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2> and
      (not constant_matrix<Arg1>) and (not constant_matrix<Arg2>) and
      (constant_diagonal_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>,
        Arg1, Arg2>>::value == 0)>>
#endif
    : constant_coefficient_type<short {0},
      typename Eigen::internal::scalar_difference_op<Scalar1, Scalar2>::result_type> {};


  /// The difference between two constant matrices is also constant.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg1> - constant_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_difference_op<Scalar1, Scalar2>::result_type> {};


  /// The difference between two constant-diagonal matrices is zero if the matrices are equal.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
  requires (not constant_matrix<Arg1>) and (not constant_matrix<Arg2>) and
    (constant_diagonal_coefficient_v<Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>,
      Arg1, Arg2>> == 0)
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2> and
      (not constant_matrix<Arg1>) and (not constant_matrix<Arg2>) and
      (constant_diagonal_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>,
        Arg1, Arg2>>::value == 0)>>
#endif
    : constant_coefficient_type<short {0},
      typename Eigen::internal::scalar_difference_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise product of two constant arrays is also constant.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg1> * constant_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_product_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise product that includes a zero matrix is zero.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2> requires
    (not constant_matrix<Arg1> and zero_matrix<Arg2>) or (zero_matrix<Arg1> and not constant_matrix<Arg2>)
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<(not constant_matrix<Arg1> and zero_matrix<Arg2>) or
      (zero_matrix<Arg1> and not constant_matrix<Arg2>)>>
#endif
    : constant_coefficient_type<short {0},
        typename Eigen::internal::scalar_difference_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise conjugate product of two constant arrays is also constant.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_conj_product_op<Scalar1, Scalar2>,
    Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_conj_product_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg1> * constant_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_conj_product_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise conjugate product that includes a zero matrix is zero.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2> requires
    (not constant_matrix<Arg1> and zero_matrix<Arg2>) or (zero_matrix<Arg1> and not constant_matrix<Arg2>)
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_conj_product_op<Scalar1, Scalar2>,
    Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_conj_product_op<Scalar1, Scalar2>,
    Arg1, Arg2>, std::enable_if_t<
      (not constant_matrix<Arg1> and zero_matrix<Arg2>) or (zero_matrix<Arg1> and not constant_matrix<Arg2>)>>
#endif
    : constant_coefficient_type<short {0},
        typename Eigen::internal::scalar_difference_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise quotient of two constant arrays is also constant.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
  requires (not zero_matrix<Arg2>)
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_quotient_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_quotient_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2> and (not zero_matrix<Arg2>)>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg1> / constant_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_quotient_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise quotient of a zero matrix and another matrix is zero.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, zero_matrix Arg1, typename Arg2> requires (not constant_matrix<Arg2>)
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_quotient_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_quotient_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<(not constant_matrix<Arg2>) and zero_matrix<Arg1>>>
#endif
    : constant_coefficient_type<short {0},
        typename Eigen::internal::scalar_difference_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise min of two constant arrays is also constant.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_min_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_min_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg1> <= constant_coefficient_v<Arg2> ?
        constant_coefficient_v<Arg1> : constant_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_min_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise max of two constant arrays is also constant.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_max_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_max_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg1> >= constant_coefficient_v<Arg2> ?
        constant_coefficient_v<Arg1> : constant_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_max_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise hypotenuse of two constant arrays is also constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<Scalar, Scalar>, Arg1, Arg2>>
#else
  template<typename Scalar, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<Scalar, Scalar>, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<
#if __cpp_nontype_template_args >= 201911L
        OpenKalman::internal::constexpr_sqrt(constant_coefficient_v<Arg1> * constant_coefficient_v<Arg1> +
          constant_coefficient_v<Arg2> * constant_coefficient_v<Arg2>)> {};
#else
        OpenKalman::internal::constexpr_sqrt(
          static_cast<short>(constant_coefficient_v<Arg1> * constant_coefficient_v<Arg1> +
          constant_coefficient_v<Arg2> * constant_coefficient_v<Arg2>))> {};
#endif


  /// The coefficient-wise power of two constant arrays is also constant.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_pow_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_pow_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<
        OpenKalman::internal::constexpr_pow(constant_coefficient_v<Arg1>, constant_coefficient_v<Arg2>),
        typename Eigen::internal::scalar_pow_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise AND of two constant arrays is also constant.
#ifdef __cpp_concepts
  template<constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_boolean_and_op, Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_boolean_and_op, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<bool {constant_coefficient_v<Arg1>} and bool {constant_coefficient_v<Arg2>}> {};


  /// The coefficient-wise OR of two constant arrays is also constant.
#ifdef __cpp_concepts
  template<constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_boolean_or_op, Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_boolean_or_op, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<bool {constant_coefficient_v<Arg1>} or bool {constant_coefficient_v<Arg2>}> {};


  /// The coefficient-wise XOR of two constant arrays is also constant.
#ifdef __cpp_concepts
  template<constant_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_boolean_xor_op, Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_boolean_xor_op, Arg1, Arg2>,
    std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<bool {constant_coefficient_v<Arg1>} xor bool {constant_coefficient_v<Arg2>}> {};


  // --- constant_diagonal_coefficient --- //


  /// The result of a constant binary operation is constant-diagonal if it is square and zero.
#ifdef __cpp_concepts
  template<typename Op, typename Arg1, typename Arg2>
  requires (not constant_diagonal_matrix<Arg1> or not constant_diagonal_matrix<Arg2>) and
    (square_matrix<Arg1> or square_matrix<Arg2>) and
    (constant_coefficient_v<Eigen::CwiseBinaryOp<Op, Arg1, Arg2>> == 0)
  struct constant_diagonal_coefficient<Eigen::CwiseBinaryOp<Op, Arg1, Arg2>>
#else
  template<typename Op, typename Arg1, typename Arg2>
  struct constant_diagonal_coefficient<Eigen::CwiseBinaryOp<Op, Arg1, Arg2>, std::enable_if_t<
    (not constant_diagonal_matrix<Arg1> or not constant_diagonal_matrix<Arg2>) and
    (square_matrix<Arg1> or square_matrix<Arg2>) and
    (constant_coefficient<Eigen::CwiseBinaryOp<Op, Arg1, Arg2>>::value == 0)>>
#endif
    : constant_coefficient<Eigen::CwiseBinaryOp<Op, Arg1, Arg2>> {};


  /// The sum of two constant-diagonal matrices is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
  struct constant_diagonal_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>,
    Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_diagonal_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>,
    Arg1, Arg2>, std::enable_if_t<constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg1> + constant_diagonal_coefficient_v<Arg2>,
      typename Eigen::internal::scalar_sum_op<Scalar1, Scalar2>::result_type> {};


  /// The difference between two constant-diagonal matrices is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
  struct constant_diagonal_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>,
    Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_diagonal_coefficient<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>, std::enable_if_t<
      constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg1> - constant_diagonal_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_difference_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise product of two constant-diagonal arrays is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
  struct constant_diagonal_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<Scalar1, Scalar2>,
    Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_diagonal_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<Scalar1, Scalar2>,
    Arg1, Arg2>,
    std::enable_if_t<constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg1> * constant_diagonal_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_product_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise conjugate product of two constant-diagonal arrays is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
  struct constant_diagonal_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_conj_product_op<Scalar1, Scalar2>,
    Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_diagonal_coefficient<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_conj_product_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg1> * constant_diagonal_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_conj_product_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise quotient of two constant-diagonal arrays is also constant if it is a one-by-one matrix.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
  requires (one_by_one_matrix<Arg1> or one_by_one_matrix<Arg2>) and (not zero_matrix<Arg2>)
  struct constant_diagonal_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_quotient_op<Scalar1, Scalar2>,
    Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_diagonal_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_quotient_op<Scalar1, Scalar2>,
    Arg1, Arg2>, std::enable_if_t<constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2> and
    (one_by_one_matrix<Arg1> or one_by_one_matrix<Arg2>) and (not zero_matrix<Arg2>)>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg1> / constant_diagonal_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_quotient_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise min of two constant-diagonal arrays is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
  struct constant_diagonal_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_min_op<Scalar1, Scalar2>,
    Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_diagonal_coefficient<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_min_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg1> <= constant_diagonal_coefficient_v<Arg2> ?
        constant_diagonal_coefficient_v<Arg1> : constant_diagonal_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_min_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise max of two constant-diagonal arrays is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
  struct constant_diagonal_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_max_op<Scalar1, Scalar2>,
    Arg1, Arg2>>
#else
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct constant_diagonal_coefficient<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_max_op<Scalar1, Scalar2>, Arg1, Arg2>,
    std::enable_if_t<constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg1> >= constant_diagonal_coefficient_v<Arg2> ?
        constant_diagonal_coefficient_v<Arg1> : constant_diagonal_coefficient_v<Arg2>,
        typename Eigen::internal::scalar_max_op<Scalar1, Scalar2>::result_type> {};


  /// The coefficient-wise hypotenuse of two constant-diagonal arrays is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
  struct constant_diagonal_coefficient<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<Scalar, Scalar>,
    Arg1, Arg2>>
#else
  template<typename Scalar, typename Arg1, typename Arg2>
  struct constant_diagonal_coefficient<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<Scalar, Scalar>, Arg1, Arg2>,
    std::enable_if_t<constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    : constant_coefficient_type<
#if __cpp_nontype_template_args >= 201911L
      OpenKalman::internal::constexpr_sqrt(
        constant_diagonal_coefficient_v<Arg1> * constant_diagonal_coefficient_v<Arg1> +
        constant_diagonal_coefficient_v<Arg2> * constant_diagonal_coefficient_v<Arg2>)> {};
#else
      OpenKalman::internal::constexpr_sqrt(static_cast<short>(
        constant_diagonal_coefficient_v<Arg1> * constant_diagonal_coefficient_v<Arg1> +
        constant_diagonal_coefficient_v<Arg2> * constant_diagonal_coefficient_v<Arg2>))> {};
#endif


  // --- is_diagonal_matrix --- //

  /// The sum of two diagonal matrices is also diagonal.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_diagonal_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<diagonal_matrix<Arg1> and diagonal_matrix<Arg2>> {};


  /// The difference between two diagonal matrices is also diagonal.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_diagonal_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<diagonal_matrix<Arg1> and diagonal_matrix<Arg2>> {};


  /// A diagonal times another array, or the product of upper and lower triangular arrays ( in either order) is diagonal.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_diagonal_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<diagonal_matrix<Arg1> or diagonal_matrix<Arg2> or
      (lower_triangular_matrix<Arg1> and upper_triangular_matrix<Arg2>) or
      (upper_triangular_matrix<Arg1> and lower_triangular_matrix<Arg2>)> {};


  // --- is_lower_self_adjoint_matrix and is_upper_self_adjoint_matrix --- //

  /// The sum of two self-adjoint matrices is lower-self-adjoint if at least one of the matrices is lower-self-adjoint.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_lower_self_adjoint_matrix<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<self_adjoint_matrix<Arg1> and self_adjoint_matrix<Arg2> and
      (lower_self_adjoint_matrix<Arg1> or lower_self_adjoint_matrix<Arg2>)> {};


  /// The sum of two upper-self-adjoint matrices is upper-self-adjoint.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_upper_self_adjoint_matrix<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<upper_self_adjoint_matrix<Arg1> and upper_self_adjoint_matrix<Arg2>> {};


  /// The difference between two self-adjoint matrices is lower-self-adjoint if at least one is lower-self-adjoint.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_lower_self_adjoint_matrix<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<self_adjoint_matrix<Arg1> and self_adjoint_matrix<Arg2> and
      (lower_self_adjoint_matrix<Arg1> or lower_self_adjoint_matrix<Arg2>)> {};


  /// The difference between two upper-self-adjoint matrices is upper-self-adjoint.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_upper_self_adjoint_matrix<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<upper_self_adjoint_matrix<Arg1> and upper_self_adjoint_matrix<Arg2>> {};


  /// The product of two self-adjoint matrices is lower-self-adjoint if at least one of the matrices is lower-self-adjoint.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_lower_self_adjoint_matrix<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<self_adjoint_matrix<Arg1> and self_adjoint_matrix<Arg2> and
      (lower_self_adjoint_matrix<Arg1> or lower_self_adjoint_matrix<Arg2>)> {};


  /// The product of two upper-self-adjoint matrices is upper-self-adjoint.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_upper_self_adjoint_matrix<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<upper_self_adjoint_matrix<Arg1> and upper_self_adjoint_matrix<Arg2>> {};


  /// A lower-triangular array times a lower-triangular array is also lower-triangular.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_lower_triangular_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<lower_triangular_matrix<Arg1> and lower_triangular_matrix<Arg2>> {};


  /// An upper-triangular array times an upper-triangular array is also upper-triangular.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_upper_triangular_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<upper_triangular_matrix<Arg1> and upper_triangular_matrix<Arg2>> {};


  // ---------------- //
  //  CwiseNullaryOp  //
  // ---------------- //

  namespace interface
  {
    template<typename UnaryOp, typename PlainObjectType>
    struct Dependencies<Eigen::CwiseNullaryOp<UnaryOp, PlainObjectType>>
    {
      static constexpr bool has_runtime_parameters =
        Eigen::internal::traits<PlainObjectType>::RowsAtCompileTime == Eigen::Dynamic or
        Eigen::internal::traits<PlainObjectType>::ColsAtCompileTime == Eigen::Dynamic;
      using type = std::tuple<>;
    };


    template<typename Scalar, typename PlainObjectType>
    struct Dependencies<Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, PlainObjectType>>
    {
      static constexpr bool has_runtime_parameters = true;
      using type = std::tuple<>;
    };


    template<typename Scalar, typename PacketType, typename PlainObjectType>
    struct Dependencies<Eigen::CwiseNullaryOp<Eigen::internal::linspaced_op<Scalar, PacketType>, PlainObjectType>>
    {
      static constexpr bool has_runtime_parameters = true;
      using type = std::tuple<>;
    };
  }


  /// \brief An Eigen nullary operation is constant if it is identity and one-by-one.
#ifdef __cpp_concepts
  template<typename Scalar, one_by_one_matrix Arg>
  struct constant_coefficient<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<Scalar>, Arg>,
      std::enable_if_t<one_by_one_matrix<Arg>>>
#endif
    : constant_coefficient_type<short {1}, Scalar> {};


  /// \brief An Eigen nullary operation is constant-diagonal if it is identity and square.
#ifdef __cpp_concepts
  template<typename Scalar, square_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<Scalar>, Arg>,
      std::enable_if_t<square_matrix<Arg>>>
#endif
    : constant_coefficient_type<short {1}, Scalar> {};


  /// A constant square matrix is lower-self-adjoint if it is not complex.
  template<typename Scalar, typename PlainObjectType>
  struct is_lower_self_adjoint_matrix<Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>,
    PlainObjectType>>
    : std::bool_constant<square_matrix<PlainObjectType> and not complex_number<Scalar>> {};


  /// A constant square matrix is upper-self-adjoint if it is not complex.
  template<typename Scalar, typename PlainObjectType>
  struct is_upper_self_adjoint_matrix<Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>,
    PlainObjectType>>
    : std::bool_constant<square_matrix<PlainObjectType> and not complex_number<Scalar>> {};


  namespace interface
  {
    // ---------------- //
    //  CwiseTernaryOp  //
    // ---------------- //

    template<typename TernaryOp, typename Arg1, typename Arg2, typename Arg3>
    struct Dependencies<Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>
    {
    private:

      using T = Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>;
      using N = Eigen::CwiseTernaryOp<TernaryOp,
        equivalent_self_contained_t<Arg1>, equivalent_self_contained_t<Arg2>, equivalent_self_contained_t<Arg3>>;

    public:

      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename T::Arg1Nested, typename T::Arg2Nested, typename T::Arg3Nested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i < 3);
        if constexpr (i == 0)
          return std::forward<Arg>(arg).arg1();
        else if constexpr (i == 1)
          return std::forward<Arg>(arg).arg2();
        else
          return std::forward<Arg>(arg).arg3();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        // Do a partial evaluation as long as at least two arguments are already self-contained.
        if constexpr (
          ((self_contained<Arg1> ? 1 : 0) + (self_contained<Arg2> ? 1 : 0) + (self_contained<Arg3> ? 1 : 0) >= 2) and
          not std::is_lvalue_reference_v<typename N::Arg1Nested> and
          not std::is_lvalue_reference_v<typename N::Arg2Nested> and
          not std::is_lvalue_reference_v<typename N::Arg3Nested>)
        {
          return N {make_self_contained(arg.arg1()), make_self_contained(arg.arg2()), make_self_contained(arg.arg3()),
                    arg.functor()};
        }
        else
        {
          return make_native_matrix(std::forward<Arg>(arg));
        }
      }
    };


    // -------------- //
    //  CwiseUnaryOp  //
    // -------------- //

    template<typename UnaryOp, typename XprType>
    struct Dependencies<Eigen::CwiseUnaryOp<UnaryOp, XprType>>
    {
    private:

      using T = Eigen::CwiseUnaryOp<UnaryOp, XprType>;
      using N = Eigen::CwiseUnaryOp<UnaryOp, equivalent_self_contained_t<XprType>>;

    public:

      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename T::XprTypeNested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        if constexpr (not std::is_lvalue_reference_v<typename N::XprTypeNested>)
          return N {make_self_contained(arg.nestedExpression()), arg.functor()};
        else
          return make_native_matrix(std::forward<Arg>(arg));
      }
    };


    template<typename BinaryOp, typename XprType>
    struct Dependencies<Eigen::CwiseUnaryOp<Eigen::internal::bind1st_op<BinaryOp>, XprType>>
    {
      static constexpr bool has_runtime_parameters = true;
      using type =
        std::tuple<typename Eigen::CwiseUnaryOp<Eigen::internal::bind1st_op<BinaryOp>, XprType>::XprTypeNested>;
    };


    template<typename BinaryOp, typename XprType>
    struct Dependencies<Eigen::CwiseUnaryOp<Eigen::internal::bind2nd_op<BinaryOp>, XprType>>
    {
      static constexpr bool has_runtime_parameters = true;
      using type =
        std::tuple<typename Eigen::CwiseUnaryOp<Eigen::internal::bind2nd_op<BinaryOp>, XprType>::XprTypeNested>;
    };

  } // namespace interface


  // --- constant_coefficient --- //


  /// The negation of a constant matrix is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
    : constant_coefficient_type<short {0} - constant_coefficient_v<Arg>> {};


  /// The conjugate of a constant matrix is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_conjugate_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_conjugate_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
    : constant_coefficient_type<
#if __cpp_nontype_template_args >= 201911L
      complex_number<Scalar> ?
#ifdef __cpp_lib_constexpr_complex
      std::conj(constant_coefficient_v<Arg>) :
#else
      std::complex(std::real(constant_coefficient_v<Arg>), -std::imag(constant_coefficient_v<Arg>)) :
#endif
#endif
      constant_coefficient_v<Arg>> {};


  /// The real part of a constant matrix is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_real_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_real_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
#if __cpp_nontype_template_args >= 201911L
    : constant_coefficient_type<std::real(constant_coefficient_v<Arg>)> {};
#else
    : constant_coefficient<std::decay_t<Arg>> {};
#endif


  /// The imaginary part of a constant matrix is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_imag_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_imag_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
#if __cpp_nontype_template_args >= 201911L
    : constant_coefficient_type<std::imag(constant_coefficient_v<Arg>)> {};
#else
    : constant_coefficient_type<short {0}, Scalar> {};
#endif


  /// The absolute value of a constant array is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_abs_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_abs_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg> >= 0 ?
      constant_coefficient_v<Arg> : -constant_coefficient_v<Arg>> {};


  /// The squared absolute value of a constant array is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_abs2_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_abs2_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg> * constant_coefficient_v<Arg>> {};


  /// The square root a constant array is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_sqrt_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_sqrt_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
    : constant_coefficient_type<OpenKalman::internal::constexpr_sqrt(constant_coefficient_v<Arg>)> {};


  /// The inverse of a constant array is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_inverse_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_inverse_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
    : constant_coefficient_type<short {1} / constant_coefficient_v<Arg>> {};


  /// The square of a constant array is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_square_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_square_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg> * constant_coefficient_v<Arg>> {};


  /// The cube of a constant array is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_cube_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_cube_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
    : constant_coefficient_type<
      constant_coefficient_v<Arg> * constant_coefficient_v<Arg> * constant_coefficient_v<Arg>> {};


  /// The logical not of a constant array is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_boolean_not_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_boolean_not_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
    : constant_coefficient_type<not bool {constant_coefficient_v<Arg>}> {};


  // --- constant_diagonal_coefficient --- //


  /// The result of a unary operation is constant-diagonal if it is square and zero.
#ifdef __cpp_concepts
  template<typename Op, typename Arg> requires (not constant_diagonal_matrix<Arg>) and
    square_matrix<Arg> and (constant_coefficient_v<Eigen::CwiseUnaryOp<Op, Arg>> == 0)
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Op, Arg>>
#else
  template<typename Op, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Op, Arg>, std::enable_if_t<
    (not constant_diagonal_matrix<Arg>) and square_matrix<Arg> and
    (constant_coefficient<Eigen::CwiseUnaryOp<Op, Arg>>::value == 0)>>
#endif
    : constant_coefficient<Eigen::CwiseUnaryOp<Op, Arg>> {};


  /// The negation of a constant-diagonal matrix is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    : constant_coefficient_type<short {0} - constant_diagonal_coefficient_v<Arg>> {};


  /// The conjugate of a constant-diagonal matrix is constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_conjugate_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_conjugate_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    : constant_coefficient_type<
#if __cpp_nontype_template_args >= 201911L
      complex_number<Scalar> ?
#ifdef __cpp_lib_constexpr_complex
      std::conj(constant_diagonal_coefficient_v<Arg>) :
#else
      std::complex(std::real(constant_diagonal_coefficient_v<Arg>), -std::imag(constant_diagonal_coefficient_v<Arg>)) :
#endif
#endif
      constant_diagonal_coefficient_v<Arg>> {};


  /// The real part of a constant-diagonal matrix is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_real_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_real_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
#if __cpp_nontype_template_args >= 201911L
    : constant_coefficient_type<std::real(constant_diagonal_coefficient_v<Arg>)> {};
#else
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg>> {};
#endif


  /// The imaginary part of a constant-diagonal matrix is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_imag_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_imag_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
#if __cpp_nontype_template_args >= 201911L
    : constant_coefficient_type<std::imag(constant_diagonal_coefficient_v<Arg>)> {};
#else
    : constant_coefficient_type<short {0}, Scalar> {};
#endif


  /// The coefficient-wise absolute value of a constant-diagonal array is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_abs_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_abs_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg> >= 0 ?
        constant_diagonal_coefficient_v<Arg> : -constant_diagonal_coefficient_v<Arg>> {};


  /// The coefficient-wise squared absolute value of a constant-diagonal array is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_abs2_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_abs2_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg> * constant_diagonal_coefficient_v<Arg>> {};


  /// The coefficient-wise square root a constant-diagonal array is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_sqrt_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_sqrt_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    : constant_coefficient_type<OpenKalman::internal::constexpr_sqrt(constant_diagonal_coefficient_v<Arg>)> {};


  /// The coefficient-wise inverse of a constant-diagonal array is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_inverse_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_inverse_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    : constant_coefficient_type<short {1} / constant_diagonal_coefficient_v<Arg>> {};


  /// The coefficient-wise square of a constant-diagonal array is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_square_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_square_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg> * constant_diagonal_coefficient_v<Arg>> {};


  /// The coefficient-wise cube of a constant-diagonal array is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_cube_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_cube_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg> * constant_diagonal_coefficient_v<Arg> *
        constant_diagonal_coefficient_v<Arg>> {};


  /// The logical not of a constant-diagonal array is constant-diagonal if it is one-by-one.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg> requires one_by_one_matrix<Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_boolean_not_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_boolean_not_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg> and one_by_one_matrix<Arg>>>
#endif
    : constant_coefficient_type<not bool {constant_diagonal_coefficient_v<Arg>}> {};


  /// The negation of a diagonal matrix is also diagonal.
  template<typename Scalar, typename Arg>
  struct is_diagonal_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, Arg>>
    : std::bool_constant<diagonal_matrix<Arg>> {};


  /// The conjugate of a diagonal matrix is also diagonal.
  template<typename Scalar, typename Arg>
  struct is_diagonal_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_conjugate_op<Scalar>, Arg>>
    : std::bool_constant<diagonal_matrix<Arg>> {};


  /// Any unary operation on a lower-self-adjoint matrix is also lower-self-adjoint.
  template<typename UnaryOp, typename Arg>
  struct is_lower_self_adjoint_matrix<Eigen::CwiseUnaryOp<UnaryOp, Arg>>
    : std::bool_constant<lower_self_adjoint_matrix<Arg>> {};


  /// Any unary operation on an upper-self-adjoint matrix is also upper-self-adjoint.
  template<typename UnaryOp, typename Arg>
  struct is_upper_self_adjoint_matrix<Eigen::CwiseUnaryOp<UnaryOp, Arg>>
    : std::bool_constant<upper_self_adjoint_matrix<Arg>> {};


  template<typename Scalar, typename Arg>
  struct is_lower_triangular_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, Arg>>
    : std::bool_constant<lower_triangular_matrix<Arg>> {};


  template<typename Scalar, typename Arg>
  struct is_lower_triangular_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_conjugate_op<Scalar>, Arg>>
    : std::bool_constant<lower_triangular_matrix<Arg>> {};


  template<typename Scalar, typename Arg>
  struct is_upper_triangular_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, Arg>>
    : std::bool_constant<upper_triangular_matrix<Arg>> {};


  template<typename Scalar, typename Arg>
  struct is_upper_triangular_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_conjugate_op<Scalar>, Arg>>
    : std::bool_constant<upper_triangular_matrix<Arg>> {};


  // ---------------- //
  //  CwiseUnaryView  //
  // ---------------- //

  namespace interface
  {
    template<typename ViewOp, typename MatrixType>
    struct Dependencies<Eigen::CwiseUnaryView<ViewOp, MatrixType>>
    {
    private:

      using T = Eigen::CwiseUnaryView<ViewOp, MatrixType>;
      using N = Eigen::CwiseUnaryView<ViewOp, equivalent_self_contained_t<MatrixType>>;

    public:

      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename T::MatrixTypeNested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        if constexpr (not std::is_lvalue_reference_v<typename N::MatrixTypeNested>)
          return N {make_self_contained(arg.nestedExpression()), arg.functor()};
        else
          return make_native_matrix(std::forward<Arg>(arg));
      }
    };
  }


  /// The real part of a constant matrix is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
#if __cpp_nontype_template_args >= 201911L
    : constant_coefficient_type<std::real(constant_coefficient_v<Arg>)> {};
#else
    : constant_coefficient<std::decay_t<Arg>> {};
#endif


  /// The imaginary part of a constant matrix is constant.
#ifdef __cpp_concepts
  template<typename Scalar, constant_matrix Arg>
  struct constant_coefficient<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_coefficient<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<Scalar>, Arg>,
    std::enable_if_t<constant_matrix<Arg>>>
#endif
#if __cpp_nontype_template_args >= 201911L
    : constant_coefficient_type<std::imag(constant_coefficient_v<Arg>)> {};
#else
    : constant_coefficient_type<short {0}, Scalar> {};
#endif


  /// A constant CwiseUnaryView is also constant-diagonal if it is square and zero.
#ifdef __cpp_concepts
  template<typename Op, typename Arg> requires (not constant_diagonal_matrix<Arg>) and square_matrix<Arg> and
    (constant_coefficient_v<Eigen::CwiseUnaryView<Op, Arg>> == 0)
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryView<Op, Arg>>
#else
  template<typename Op, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryView<Op, Arg>, std::enable_if_t<
    (not constant_diagonal_matrix<Arg>) and square_matrix<Arg> and
    (constant_coefficient<Eigen::CwiseUnaryView<Op, Arg>>::value == 0)>>
#endif
    : constant_coefficient<Eigen::CwiseUnaryView<Op, Arg>> {};


  /// The real part of a constant-diagonal matrix is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
#if __cpp_nontype_template_args >= 201911L
    : constant_coefficient_type<std::real(constant_diagonal_coefficient_v<Arg>)> {};
#else
    : constant_diagonal_coefficient<std::decay_t<Arg>> {};
#endif


  /// The imaginary part of a constant-diagonal matrix is also constant-diagonal.
#ifdef __cpp_concepts
  template<typename Scalar, constant_diagonal_matrix Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<Scalar>, Arg>>
#else
  template<typename Scalar, typename Arg>
  struct constant_diagonal_coefficient<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<Scalar>, Arg>,
    std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
#if __cpp_nontype_template_args >= 201911L
    : constant_coefficient_type<std::imag(constant_diagonal_coefficient_v<Arg>)> {};
#else
    : constant_coefficient_type<short {0}, Scalar> {};
#endif


  /// The real part of a diagonal matrix is also diagonal.
  template<typename Scalar, typename Arg>
  struct is_diagonal_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<Scalar>, Arg>>
    : std::bool_constant<diagonal_matrix<Arg>> {};


  /// The imaginary part of a diagonal matrix is also diagonal.
  template<typename Scalar, typename Arg>
  struct is_diagonal_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<Scalar>, Arg>>
    : std::bool_constant<diagonal_matrix<Arg>> {};


  /// The real part of a lower-self-adjoint matrix is also lower-self-adjoint.
  template<typename Scalar, typename Arg>
  struct is_lower_self_adjoint_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<Scalar>, Arg>>
    : std::bool_constant<lower_self_adjoint_matrix<Arg>> {};


  /// The imaginary part of a lower-self-adjoint matrix is also lower-self-adjoint.
  template<typename Scalar, typename Arg>
  struct is_lower_self_adjoint_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<Scalar>, Arg>>
    : std::bool_constant<lower_self_adjoint_matrix<Arg>> {};


  /// The real part of an upper-self-adjoint matrix is also upper-self-adjoint.
  template<typename Scalar, typename Arg>
  struct is_upper_self_adjoint_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<Scalar>, Arg>>
    : std::bool_constant<upper_self_adjoint_matrix<Arg>> {};


  /// The imaginary part of an upper-self-adjoint matrix is also upper-self-adjoint.
  template<typename Scalar, typename Arg>
  struct is_upper_self_adjoint_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<Scalar>, Arg>>
    : std::bool_constant<upper_self_adjoint_matrix<Arg>> {};


  /// The real part of a lower-triangular matrix is also lower-triangular.
  template<typename Scalar, typename Arg>
  struct is_lower_triangular_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<Scalar>, Arg>>
    : std::bool_constant<lower_triangular_matrix<Arg>> {};


  /// The imaginary part of a lower-triangular matrix is also lower-triangular.
  template<typename Scalar, typename Arg>
  struct is_lower_triangular_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<Scalar>, Arg>>
    : std::bool_constant<lower_triangular_matrix<Arg>> {};


  /// The real part of an upper-triangular matrix is also upper-triangular.
  template<typename Scalar, typename Arg>
  struct is_upper_triangular_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_real_ref_op<Scalar>, Arg>>
    : std::bool_constant<upper_triangular_matrix<Arg>> {};


  /// The imaginary part of an upper-triangular matrix is also upper-triangular.
  template<typename Scalar, typename Arg>
  struct is_upper_triangular_matrix<Eigen::CwiseUnaryView<Eigen::internal::scalar_imag_ref_op<Scalar>, Arg>>
    : std::bool_constant<upper_triangular_matrix<Arg>> {};


  // ---------- //
  //  Diagonal  //
  // ---------- //

  namespace interface
  {
    template<typename MatrixType, int DiagIndex>
    struct Dependencies<Eigen::Diagonal<MatrixType, DiagIndex>>
    {
      static constexpr bool has_runtime_parameters = DiagIndex == Eigen::DynamicIndex;
      using type = std::tuple<typename Eigen::internal::ref_selector<MatrixType>::non_const_type>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      // Should always convert to a dense, writable matrix.

    };
  }


  namespace detail
  {
    template<typename T> struct is_eigen_identity : std::false_type {};

    template<typename Scalar, typename Arg>
    struct is_eigen_identity<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<Scalar>, Arg>>
      : std::true_type {};

#ifdef __cpp_concepts
    template<typename T> concept eigen_identity = is_eigen_identity<std::decay_t<T>>::value;
#else
    template<typename T> static constexpr bool eigen_identity = is_eigen_identity<std::decay_t<T>>::value;
#endif
  }


  /// The main diagonal of an identity matrix is constant 1; other diagonals are constant 0.
#ifdef __cpp_concepts
  template<OpenKalman::detail::eigen_identity MatrixType, int DiagIndex> requires (DiagIndex != Eigen::DynamicIndex)
  struct constant_coefficient<Eigen::Diagonal<MatrixType, DiagIndex>>
#else
  template<typename MatrixType, int DiagIndex>
  struct constant_coefficient<Eigen::Diagonal<MatrixType, DiagIndex>, std::enable_if_t<
    OpenKalman::detail::eigen_identity<MatrixType> and DiagIndex != Eigen::DynamicIndex>>
#endif
    : constant_coefficient_type<short {DiagIndex == 0 ? 1 : 0}, scalar_type_of_t<MatrixType>> {};


  /// The diagonal of a constant matrix is constant.
#ifdef __cpp_concepts
  template<constant_matrix MatrixType, int DiagIndex> requires (not OpenKalman::detail::eigen_identity<MatrixType>)
  struct constant_coefficient<Eigen::Diagonal<MatrixType, DiagIndex>>
#else
  template<typename MatrixType, int DiagIndex>
  struct constant_coefficient<Eigen::Diagonal<MatrixType, DiagIndex>, std::enable_if_t<
    constant_matrix<MatrixType> and not OpenKalman::detail::eigen_identity<MatrixType>>>
#endif
    : constant_coefficient<std::decay_t<MatrixType>> {};


  /// The main diagonal of a constant-diagonal matrix is constant; other diagonals are constant 0.
#ifdef __cpp_concepts
  template<constant_diagonal_matrix MatrixType, int DiagIndex> requires (not constant_matrix<MatrixType>) and
    (not OpenKalman::detail::eigen_identity<MatrixType>)
  struct constant_coefficient<Eigen::Diagonal<MatrixType, DiagIndex>>
#else
  template<typename MatrixType, int DiagIndex>
  struct constant_coefficient<Eigen::Diagonal<MatrixType, DiagIndex>, std::enable_if_t<
    constant_diagonal_matrix<MatrixType> and (not constant_matrix<MatrixType>) and
    (not OpenKalman::detail::eigen_identity<MatrixType>)>>
#endif
    : constant_coefficient_type<(DiagIndex == 0 ? constant_diagonal_coefficient_v<std::decay_t<MatrixType>> : 0),
      scalar_type_of_t<MatrixType>> {};


  /// The diagonal of a constant, one-by-one matrix is constant-diagonal.
#ifdef __cpp_concepts
  template<constant_matrix MatrixType, int DiagIndex> requires (DiagIndex != Eigen::DynamicIndex) and
    (not dynamic_rows<MatrixType> or (not dynamic_columns<MatrixType> and column_extent_of_v<MatrixType> == 1)) and
    (not dynamic_columns<MatrixType> or (not dynamic_rows<MatrixType> and row_extent_of_v<MatrixType> == 1)) and
    (dynamic_shape<MatrixType> or std::min(row_extent_of_v<MatrixType> + std::min(DiagIndex, 0),
        column_extent_of_v<MatrixType> - std::max(DiagIndex, 0)) == 1)
  struct constant_diagonal_coefficient<Eigen::Diagonal<MatrixType, DiagIndex>>
#else
  template<typename MatrixType, int DiagIndex>
  struct constant_diagonal_coefficient<Eigen::Diagonal<MatrixType, DiagIndex>, std::enable_if_t<
    constant_matrix<MatrixType> and (DiagIndex != Eigen::DynamicIndex) and
    (not dynamic_rows<MatrixType> or (not dynamic_columns<MatrixType> and column_extent_of<MatrixType>::value == 1)) and
    (not dynamic_columns<MatrixType> or (not dynamic_rows<MatrixType> and row_extent_of<MatrixType>::value == 1)) and
    (dynamic_shape<MatrixType> or std::min(row_extent_of<MatrixType>::value + std::min(DiagIndex, 0),
        column_extent_of<MatrixType>::value - std::max(DiagIndex, 0)) == 1)>>
#endif
    : constant_coefficient<std::decay_t<MatrixType>> {};


  /// The diagonal of a constant-diagonal matrix is constant-diagonal if the result is one-by-one
#ifdef __cpp_concepts
  template<constant_diagonal_matrix MatrixType, int DiagIndex>
  requires (not constant_matrix<MatrixType>) and (DiagIndex != Eigen::DynamicIndex) and
    (not dynamic_rows<MatrixType> or (not dynamic_columns<MatrixType> and column_extent_of_v<MatrixType> == 1)) and
    (not dynamic_columns<MatrixType> or (not dynamic_rows<MatrixType> and row_extent_of_v<MatrixType> == 1)) and
    (dynamic_shape<MatrixType> or std::min(row_extent_of_v<MatrixType> + std::min(DiagIndex, 0),
        column_extent_of_v<MatrixType> - std::max(DiagIndex, 0)) == 1)
  struct constant_diagonal_coefficient<Eigen::Diagonal<MatrixType, DiagIndex>>
#else
  template<typename MatrixType, int DiagIndex>
  struct constant_diagonal_coefficient<Eigen::Diagonal<MatrixType, DiagIndex>, std::enable_if_t<
    constant_diagonal_matrix<MatrixType> and (not constant_matrix<MatrixType>) and
    (DiagIndex != Eigen::DynamicIndex) and
    (not dynamic_rows<MatrixType> or (not dynamic_columns<MatrixType> and column_extent_of<MatrixType>::value == 1)) and
    (not dynamic_columns<MatrixType> or (not dynamic_rows<MatrixType> and row_extent_of<MatrixType>::value == 1)) and
    (dynamic_shape<MatrixType> or std::min(row_extent_of<MatrixType>::value + std::min(DiagIndex, 0),
        column_extent_of<MatrixType>::value - std::max(DiagIndex, 0)) == 1)>>
#endif
    : constant_coefficient_type<(DiagIndex == 0 ? constant_diagonal_coefficient_v<std::decay_t<MatrixType>> : 0),
      scalar_type_of_t<MatrixType>> {};


  // ---------------- //
  //  DiagonalMatrix  //
  // ---------------- //


  namespace interface
  {
    template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
    struct Dependencies<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<
        typename Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>::DiagonalVectorType>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).diagonal();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        return make_self_contained(Eigen3::DiagonalMatrix {std::forward<Arg>(arg)});
      }
    };
  } // namespace interface


  /**
   * \internal
   * \brief Matrix traits for Eigen::DiagonalMatrix.
   */
  template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
  struct MatrixTraits<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    : MatrixTraits<Eigen::Matrix<Scalar, SizeAtCompileTime, SizeAtCompileTime>>
  {
  };


  template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
  struct is_diagonal_matrix<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    : std::true_type {};


  // ----------------- //
  //  DiagonalWrapper  //
  // ----------------- //

  namespace interface
  {

    template<typename DiagVectorType>
    struct Dependencies<Eigen::DiagonalWrapper<DiagVectorType>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename DiagVectorType::Nested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).diagonal();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        return make_self_contained(Eigen3::DiagonalMatrix {std::forward<Arg>(arg)});
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
        V::SizeAtCompileTime, V::SizeAtCompileTime>>
  {
  };


  /// A diagonal wrapper is constant if its nested vector is constant and is either zero or has one row.
#ifdef __cpp_concepts
  template<constant_matrix DiagonalVectorType> requires (constant_coefficient_v<DiagonalVectorType> == 0) or
    (row_extent_of_v<DiagonalVectorType> == 1)
  struct constant_coefficient<Eigen::DiagonalWrapper<DiagonalVectorType>>
#else
  template<typename DiagonalVectorType>
  struct constant_coefficient<Eigen::DiagonalWrapper<DiagonalVectorType>, std::enable_if_t<
    constant_matrix<DiagonalVectorType> and (constant_coefficient<DiagonalVectorType>::value == 0 or
    (row_extent_of<DiagonalVectorType>::value == 1))>>
#endif
    : constant_coefficient<std::decay_t<DiagonalVectorType>> {};


  /// A diagonal wrapper is constant-diagonal if its nested vector is constant.
#ifdef __cpp_concepts
  template<constant_matrix DiagonalVectorType>
  struct constant_diagonal_coefficient<Eigen::DiagonalWrapper<DiagonalVectorType>>
#else
  template<typename DiagonalVectorType>
  struct constant_diagonal_coefficient<Eigen::DiagonalWrapper<DiagonalVectorType>, std::enable_if_t<
    constant_matrix<DiagonalVectorType>>>
#endif
    : constant_coefficient<std::decay_t<DiagonalVectorType>> {};


  template<typename DiagonalVectorType>
  struct is_diagonal_matrix<Eigen::DiagonalWrapper<DiagonalVectorType>>
    : std::true_type {};


  // ------------------------- //
  //  IndexedView (Eigen 3.4)  //
  // ------------------------- //


  // ------------- //
  //  Homogeneous  //
  // ------------- //
  // \todo: Add. This is a child of Eigen::MatrixBase


  namespace interface
  {

    // --------- //
    //  Inverse  //
    // --------- //

    template<typename XprType>
    struct Dependencies<Eigen::Inverse<XprType>>
    {
    private:

      using T = Eigen::Inverse<XprType>;
      using N = Eigen::Inverse<equivalent_self_contained_t<XprType>>;

    public:

      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename T::XprTypeNested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        if constexpr (not std::is_lvalue_reference_v<N::XprTypeNested>)
          return N {make_self_contained(arg.nestedExpression())};
        else
          return make_native_matrix(std::forward<Arg>(arg));
      }
    };


    // ----- //
    //  Map  //
    // ----- //

    template<typename PlainObjectType, int MapOptions, typename StrideType>
    struct Dependencies<Eigen::Map<PlainObjectType, MapOptions, StrideType>>
    {
      static constexpr bool has_runtime_parameters = false;
      // Map is not self-contained in any circumstances.
    };


    // -------- //
    //  Matrix  //
    // -------- //

    template<typename S, int rows, int cols, int options, int maxrows, int maxcols>
    struct Dependencies<Eigen::Matrix<S, rows, cols, options, maxrows, maxcols>>
    {
      static constexpr bool has_runtime_parameters = true;
      using type = std::tuple<>;
    };


    // --------------- //
    //  MatrixWrapper  //
    // --------------- //

    template<typename XprType>
    struct Dependencies<Eigen::MatrixWrapper<XprType>>
    {
    private:

      using T = Eigen::MatrixWrapper<XprType>;
      using N = Eigen::MatrixWrapper<equivalent_self_contained_t<XprType>>;

    public:

      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename T::NestedExpressionType>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        if constexpr (not std::is_lvalue_reference_v<typename N::NestedExpressionType>)
          return N {make_self_contained(arg.nestedExpression())};
        else
          return make_native_matrix(std::forward<Arg>(arg));
      }
    };


  } // namespace interface


  /// A matrix wrapper is constant if its nested expression is constant.
#ifdef __cpp_concepts
  template<constant_matrix XprType>
  struct constant_coefficient<Eigen::MatrixWrapper<XprType>>
#else
  template<typename XprType>
  struct constant_coefficient<Eigen::MatrixWrapper<XprType>, std::enable_if_t<constant_matrix<XprType>>>
#endif
    : constant_coefficient<std::decay_t<XprType>> {};


  /// A matrix wrapper is constant-diagonal if its nested expression is constant-diagonal.
#ifdef __cpp_concepts
  template<constant_diagonal_matrix XprType>
  struct constant_diagonal_coefficient<Eigen::MatrixWrapper<XprType>>
#else
  template<typename XprType>
  struct constant_diagonal_coefficient<Eigen::MatrixWrapper<XprType>, std::enable_if_t<
    constant_diagonal_matrix<XprType>>>
#endif
    : constant_diagonal_coefficient<std::decay_t<XprType>> {};


  template<typename XprType>
  struct is_diagonal_matrix<Eigen::MatrixWrapper<XprType>>
    : std::bool_constant<diagonal_matrix<XprType>> {};


  template<typename XprType>
  struct is_lower_self_adjoint_matrix<Eigen::MatrixWrapper<XprType>>
    : std::bool_constant<lower_self_adjoint_matrix<XprType>> {};


  template<typename XprType>
  struct is_upper_self_adjoint_matrix<Eigen::MatrixWrapper<XprType>>
    : std::bool_constant<upper_self_adjoint_matrix<XprType>> {};


  template<typename XprType>
  struct is_lower_triangular_matrix<Eigen::MatrixWrapper<XprType>>
    : std::bool_constant<lower_triangular_matrix<XprType>> {};


  template<typename XprType>
  struct is_upper_triangular_matrix<Eigen::MatrixWrapper<XprType>>
    : std::bool_constant<upper_triangular_matrix<XprType>> {};


  // ------------------ //
  //  PartialReduxExpr  //
  // ------------------ //

  namespace interface
  {

    template<typename MatrixType, typename MemberOp, int Direction>
    struct Dependencies<Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename MatrixType::Nested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        using NewMatrixType = equivalent_self_contained_t<MatrixType>;
        using N = Eigen::PartialReduxExpr<NewMatrixType, MemberOp, Direction>;
        if constexpr (not std::is_lvalue_reference_v<typename NewMatrixType::Nested>)
          return N {make_self_contained(arg.nestedExpression()), arg.functor()};
        else
          return make_native_matrix(std::forward<Arg>(arg));
      }
    };

  } // namespace interface


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, int p, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_lpnorm<p, Scalar>, Direction>>
#else
  template<typename MatrixType, int p, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_lpnorm<p, Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<
      (constant_coefficient_v<MatrixType> >= 0) ?
      constant_coefficient_v<MatrixType> :
      -constant_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_squaredNorm<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_squaredNorm<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<
      (constant_coefficient_v<MatrixType> >= 0) ?
      constant_coefficient_v<MatrixType> :
      -constant_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_norm<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_norm<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<
      (constant_coefficient_v<MatrixType> >= 0) ?
      constant_coefficient_v<MatrixType> :
      -constant_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_stableNorm<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_stableNorm<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<
      (constant_coefficient_v<MatrixType> >= 0) ?
      constant_coefficient_v<MatrixType> :
      -constant_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_hypotNorm<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_hypotNorm<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<
      (constant_coefficient_v<MatrixType> >= 0) ?
      constant_coefficient_v<MatrixType> :
      -constant_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  requires (Direction == Eigen::Horizontal or not dynamic_rows<MatrixType>) and
    (Direction == Eigen::Vertical or not dynamic_columns<MatrixType>)
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_sum<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_sum<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType> and
      (Direction == Eigen::Horizontal or not dynamic_rows<MatrixType>) and
      (Direction == Eigen::Vertical or not dynamic_columns<MatrixType>)>>
#endif
    : constant_coefficient_type<constant_coefficient_v<MatrixType> *
      (Direction == Eigen::Vertical ? row_extent_of_v<MatrixType> : column_extent_of_v<MatrixType>),
      std::common_type_t<scalar_type_of_t<MatrixType>, Scalar>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_mean<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_mean<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_minCoeff<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_minCoeff<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_maxCoeff<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_maxCoeff<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_all<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_all<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_any<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_any<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Index, int Direction>
  requires (Direction == Eigen::Horizontal or not dynamic_rows<MatrixType>) and
    (Direction == Eigen::Vertical or not dynamic_columns<MatrixType>)
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_count<Index>, Direction>>
#else
  template<typename MatrixType, typename Index, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_count<Index>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType> and
      (Direction == Eigen::Horizontal or not dynamic_rows<MatrixType>) and
      (Direction == Eigen::Vertical or not dynamic_columns<MatrixType>)>>
#endif
    : constant_coefficient_type<
      Direction == Eigen::Vertical ? row_extent_of_v<MatrixType> : column_extent_of_v<MatrixType>,
      scalar_type_of_t<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename Scalar, int Direction>
  requires (Direction == Eigen::Horizontal or not dynamic_rows<MatrixType>) and
    (Direction == Eigen::Vertical or not dynamic_columns<MatrixType>)
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_prod<Scalar>, Direction>>
#else
  template<typename MatrixType, typename Scalar, int Direction>
  struct constant_coefficient<Eigen::PartialReduxExpr<MatrixType, Eigen::internal::member_prod<Scalar>, Direction>,
    std::enable_if_t<constant_matrix<MatrixType> and
      (Direction == Eigen::Horizontal or not dynamic_rows<MatrixType>) and
      (Direction == Eigen::Vertical or not dynamic_columns<MatrixType>)>>
#endif
    : constant_coefficient_type<OpenKalman::internal::constexpr_pow(constant_coefficient_v<MatrixType>,
      (Direction == Eigen::Vertical ? row_extent_of_v<MatrixType> : column_extent_of_v<MatrixType>)),
      std::common_type_t<scalar_type_of_t<MatrixType>, Scalar>> {};


  // A constant partial redux expression is constant-diagonal if it is one-by-one.
#ifdef __cpp_concepts
  template<constant_matrix MatrixType, typename MemberOp, int Direction>
  requires (Direction == Eigen::Vertical and column_extent_of_v<MatrixType> == 1) or
    (Direction == Eigen::Horizontal and row_extent_of_v<MatrixType> == 1)
  struct constant_diagonal_coefficient<Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>>
#else
  template<typename MatrixType, typename MemberOp, int Direction>
  struct constant_diagonal_coefficient<Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>,
    std::enable_if_t<constant_matrix<MatrixType> and
      ((Direction == Eigen::Vertical and column_extent_of<MatrixType>::value == 1) or
        (Direction == Eigen::Horizontal and row_extent_of<MatrixType>::value == 1))>>
#endif
    : constant_coefficient<std::decay_t<Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>>> {};


  // ------------------- //
  //  PermutationMatrix  //
  // ------------------- //

  namespace interface
  {
    template<int SizeAtCompileTime, int MaxSizeAtCompileTime, typename StorageIndex>
    struct Dependencies<Eigen::PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<
        typename Eigen::PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex>::IndicesType>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).indices();
      }

      // PermutationMatrix is always self-contained.

    };


  // -------------------- //
  //  PermutationWrapper  //
  // -------------------- //

    template<typename IndicesType>
    struct Dependencies<Eigen::PermutationWrapper<IndicesType>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename IndicesType::Nested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).indices();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        using NewIndicesType = equivalent_self_contained_t<IndicesType>;
        if constexpr (not std::is_lvalue_reference_v<typename NewIndicesType::Nested>)
          return Eigen::PermutationWrapper<NewIndicesType> {make_self_contained(arg.nestedExpression()), arg.functor()};
        else
          return make_native_matrix(std::forward<Arg>(arg));
      }
    };


  // --------- //
  //  Product  //
  // --------- //

    template<typename LhsType, typename RhsType, int Option>
    struct Dependencies<Eigen::Product<LhsType, RhsType, Option>>
    {
    private:

      using T = Eigen::Product<LhsType, RhsType>;
      using N = Eigen::Product<equivalent_self_contained_t<LhsType>, equivalent_self_contained_t<RhsType>, Option>;

    public:

      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename T::LhsNested, typename T::RhsNested >;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i < 2);
        if constexpr (i == 0)
          return std::forward<Arg>(arg).lhs();
        else
          return std::forward<Arg>(arg).rhs();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        constexpr auto to_be_evaluated_size = self_contained<LhsType> ?
          RhsType::RowsAtCompileTime * RhsType::ColsAtCompileTime :
          LhsType::RowsAtCompileTime * LhsType::ColsAtCompileTime;

        // Do a partial evaluation if at least one argument is self-contained and result size > non-self-contained size.
        if constexpr ((self_contained<LhsType> or self_contained<RhsType>) and
          (LhsType::RowsAtCompileTime != Eigen::Dynamic) and
          (LhsType::ColsAtCompileTime != Eigen::Dynamic) and
          (RhsType::RowsAtCompileTime != Eigen::Dynamic) and
          (RhsType::ColsAtCompileTime != Eigen::Dynamic) and
          (LhsType::RowsAtCompileTime * RhsType::ColsAtCompileTime > to_be_evaluated_size) and
          not std::is_lvalue_reference_v<typename N::LhsNested> and
          not std::is_lvalue_reference_v<typename N::RhsNested>)
        {
          return N {make_self_contained(arg.lhs()), make_self_contained(arg.rhs())};
        }
        else
        {
          return make_native_matrix(std::forward<Arg>(arg));
        }
      }
    };

  } // namespace interface


  /// A constant-diagonal matrix times a constant matrix is constant.
#ifdef __cpp_concepts
  template<constant_diagonal_matrix Arg1, constant_matrix Arg2>
  struct constant_coefficient<Eigen::Product<Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
    constant_diagonal_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg1> * constant_coefficient_v<Arg2>> {};


  /// A constant matrix times a constant-diagonal matrix is constant.
#ifdef __cpp_concepts
  template<constant_matrix Arg1, constant_diagonal_matrix Arg2>
  requires (not constant_diagonal_matrix<Arg1>) or (not constant_matrix<Arg2>)
  struct constant_coefficient<Eigen::Product<Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
    constant_matrix<Arg1> and constant_diagonal_matrix<Arg2> and
    (not constant_diagonal_matrix<Arg1> or not constant_matrix<Arg2>)>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg1> * constant_diagonal_coefficient_v<Arg2>> {};


  /// The product of two constant matrices is constant if the rows of A or columns of B is known at compile time.
#ifdef __cpp_concepts
  template<constant_matrix Arg1, constant_matrix Arg2>
  requires (not constant_diagonal_matrix<Arg1>) and (not constant_diagonal_matrix<Arg2>) and
    (not dynamic_columns<Arg1> or not dynamic_rows<Arg2> or
      constant_coefficient_v<Arg1> == 0 or constant_coefficient_v<Arg2> == 0)
  struct constant_coefficient<Eigen::Product<Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
    constant_matrix<Arg1> and constant_matrix<Arg2> and
    (not constant_diagonal_matrix<Arg1>) and (not constant_diagonal_matrix<Arg2>) and
    ((not dynamic_columns<Arg1>) or (not dynamic_rows<Arg2>) or
      (constant_coefficient<Arg1>::value == 0) or (constant_coefficient<Arg2>::value == 0))>>
#endif
    : constant_coefficient_type<constant_coefficient_v<Arg1> * constant_coefficient_v<Arg2> *
      int {dynamic_rows<Arg2> ? column_extent_of_v<Arg1> : row_extent_of_v<Arg2>},
      typename Eigen::Product<Arg1, Arg2>::Scalar> {};


  /// A product that includes a zero matrix is zero.
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2>
  requires (zero_matrix<Arg1> and not constant_matrix<Arg2> and not constant_diagonal_matrix<Arg2>) or
    (zero_matrix<Arg2> and not constant_matrix<Arg1> and not constant_diagonal_matrix<Arg1>)
  struct constant_coefficient<Eigen::Product<Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct constant_coefficient<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
    (zero_matrix<Arg1> and not constant_matrix<Arg2> and not constant_diagonal_matrix<Arg2>) or
    (zero_matrix<Arg2> and not constant_matrix<Arg1> and not constant_diagonal_matrix<Arg1>)>>
#endif
    : constant_coefficient_type<short {0}, typename Eigen::Product<Arg1, Arg2>::Scalar> {};


  /// A constant product of two matrices is constant-diagonal if it is square and zero or one-by-one
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    (not constant_diagonal_matrix<Arg1> or not constant_diagonal_matrix<Arg2>) and
    constant_matrix<Eigen::Product<Arg1, Arg2>> and
    (row_extent_of_v<Arg1> == column_extent_of_v<Arg2>) and
    (constant_coefficient_v<Eigen::Product<Arg1, Arg2>> == 0 or
      row_extent_of_v<Arg1> == 1 or column_extent_of_v<Arg2> == 1)
  struct constant_diagonal_coefficient<Eigen::Product<Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct constant_diagonal_coefficient<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
    (not constant_diagonal_matrix<Arg1> or not constant_diagonal_matrix<Arg2>) and
    constant_matrix<Eigen::Product<Arg1, Arg2>> and
    (row_extent_of<Arg1>::value == column_extent_of<Arg2>::value) and
    (constant_coefficient<Eigen::Product<Arg1, Arg2>>::value == 0 or
      row_extent_of<Arg1>::value == 1 or column_extent_of<Arg2>::value == 1)>>
#endif
    : constant_coefficient<Eigen::Product<Arg1, Arg2>> {};


  /// A constant-diagonal matrix times another constant-diagonal matrix is constant-diagonal.
#ifdef __cpp_concepts
  template<constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
  struct constant_diagonal_coefficient<Eigen::Product<Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct constant_diagonal_coefficient<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
    constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<Arg1> * constant_diagonal_coefficient_v<Arg2>> {};


  /// The product of two diagonal matrices is also diagonal.
  template<typename Arg1, typename Arg2>
  struct is_diagonal_matrix<Eigen::Product<Arg1, Arg2>>
    : std::bool_constant<diagonal_matrix<Arg1> and diagonal_matrix<Arg2>> {};


  /// A constant-diagonal matrix times a lower-self-adjoint matrix (or vice versa) is lower-self-adjoint.
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2>
  requires (constant_diagonal_matrix<Arg1> and lower_self_adjoint_matrix<Arg2>) or
    (lower_self_adjoint_matrix<Arg1> and constant_diagonal_matrix<Arg2>)
  struct is_lower_self_adjoint_matrix<Eigen::Product<Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct is_lower_self_adjoint_matrix<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
    (constant_diagonal_matrix<Arg1> and lower_self_adjoint_matrix<Arg2>) or
    (lower_self_adjoint_matrix<Arg1> and constant_diagonal_matrix<Arg2>)>>
#endif
    : std::true_type {};


  /// A constant-diagonal matrix times an upper-self-adjoint matrix (or vice versa) is upper-self-adjoint.
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2>
  requires (constant_diagonal_matrix<Arg1> and upper_self_adjoint_matrix<Arg2>) or
    (upper_self_adjoint_matrix<Arg1> and constant_diagonal_matrix<Arg2>)
  struct is_upper_self_adjoint_matrix<Eigen::Product<Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct is_upper_self_adjoint_matrix<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
    (constant_diagonal_matrix<Arg1> and upper_self_adjoint_matrix<Arg2>) or
    (upper_self_adjoint_matrix<Arg1> and constant_diagonal_matrix<Arg2>)>>
#endif
    : std::true_type {};


  /// A constant-diagonal matrix times an upper-triangular matrix (or vice versa) is upper-triangular.
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires (constant_diagonal_matrix<Arg1> and upper_triangular_matrix<Arg2>) or
    (upper_triangular_matrix<Arg1> and constant_diagonal_matrix<Arg2>)
  struct is_upper_triangular_matrix<Eigen::Product<Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct is_upper_triangular_matrix<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
    (constant_diagonal_matrix<Arg1> and upper_triangular_matrix<Arg2>) or
    (upper_triangular_matrix<Arg1> and constant_diagonal_matrix<Arg2>)>>
#endif
    : std::true_type {};


  /// A constant-diagonal matrix times an lower-triangular matrix (or vice versa) is lower-triangular.
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires (constant_diagonal_matrix<Arg1> and lower_triangular_matrix<Arg2>) or
    (lower_triangular_matrix<Arg1> and constant_diagonal_matrix<Arg2>)
  struct is_lower_triangular_matrix<Eigen::Product<Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct is_lower_triangular_matrix<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
    (constant_diagonal_matrix<Arg1> and lower_triangular_matrix<Arg2>) or
    (lower_triangular_matrix<Arg1> and constant_diagonal_matrix<Arg2>)>>
#endif
    : std::true_type {};


  // ----- //
  //  Ref  //
  // ----- //

  namespace interface
  {
    template<typename PlainObjectType, int Options, typename StrideType>
    struct Dependencies<Eigen::Ref<PlainObjectType, Options, StrideType>>
    {
      static constexpr bool has_runtime_parameters = false;
      // Ref is not self-contained in any circumstances.
    };


  // ----------- //
  //  Replicate  //
  // ----------- //

    template<typename MatrixType, int RowFactor, int ColFactor>
    struct Dependencies<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
    {
    private:

      using T = Eigen::Replicate<MatrixType, RowFactor, ColFactor>;
      using N = Eigen::Replicate<equivalent_self_contained_t<MatrixType>, RowFactor, ColFactor>;

    public:

      static constexpr bool has_runtime_parameters = RowFactor == Eigen::Dynamic or ColFactor == Eigen::Dynamic;
      using type = std::tuple<typename Eigen::internal::traits<T>::MatrixTypeNested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        if constexpr (not std::is_lvalue_reference_v<typename Eigen::internal::traits<N>::MatrixTypeNested>)
          return N {make_self_contained(arg.nestedExpression())};
        else
          return make_native_matrix(std::forward<Arg>(arg));
      }
    };

  } // namespace interface


  /// A replication of a constant matrix is constant.
#ifdef __cpp_concepts
  template<constant_matrix MatrixType, int RowFactor, int ColFactor>
  struct constant_coefficient<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
#else
  template<typename MatrixType, int RowFactor, int ColFactor>
  struct constant_coefficient<Eigen::Replicate<MatrixType, RowFactor, ColFactor>, std::enable_if_t<
    constant_matrix<MatrixType>>>
#endif
    : constant_coefficient<std::decay_t<MatrixType>> {};


  /// A replication of a constant matrix is constant-diagonal if the result is square and zero.
#ifdef __cpp_concepts
  template<typename MatrixType, int RowFactor, int ColFactor> requires (not constant_diagonal_matrix<MatrixType>) and
    (RowFactor > 0) and (ColFactor > 0) and (not dynamic_rows<MatrixType>) and (not dynamic_columns<MatrixType>) and
    (row_extent_of_v<MatrixType> * RowFactor == column_extent_of_v<MatrixType> * ColFactor) and
    (constant_coefficient_v<MatrixType> == 0)
  struct constant_diagonal_coefficient<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
#else
  template<typename MatrixType, int RowFactor, int ColFactor>
  struct constant_diagonal_coefficient<Eigen::Replicate<MatrixType, RowFactor, ColFactor>, std::enable_if_t<
    (not constant_diagonal_matrix<MatrixType>) and
    (RowFactor > 0) and (ColFactor > 0) and (not dynamic_rows<MatrixType>) and (not dynamic_columns<MatrixType>) and
    (row_extent_of<MatrixType>::value * RowFactor == column_extent_of<MatrixType>::value * ColFactor) and
    (zero_matrix<MatrixType>)>>
#endif
    : constant_coefficient<std::decay_t<MatrixType>> {};


  /// A replication of a constant-diagonal matrix is constant-diagonal if it is replicated only once, or is zero and becomes square.
#ifdef __cpp_concepts
  template<constant_diagonal_matrix MatrixType, int RowFactor, int ColFactor>
  requires (RowFactor == 1 and ColFactor == 1) or
    (constant_diagonal_coefficient_v<MatrixType> == 0 and RowFactor == ColFactor and RowFactor > 0)
  struct constant_diagonal_coefficient<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
#else
  template<typename MatrixType, int RowFactor, int ColFactor>
  struct constant_diagonal_coefficient<Eigen::Replicate<MatrixType, RowFactor, ColFactor>, std::enable_if_t<
    constant_diagonal_matrix<MatrixType> and ((RowFactor == 1 and ColFactor == 1) or
    (constant_diagonal_coefficient<MatrixType>::value == 0 and RowFactor == ColFactor and RowFactor > 0))>>
#endif
    : constant_diagonal_coefficient<std::decay_t<MatrixType>> {};


  template<typename MatrixType>
  struct is_diagonal_matrix<Eigen::Replicate<MatrixType, 1, 1>>
    : std::bool_constant<diagonal_matrix<MatrixType>> {};


  template<typename MatrixType>
  struct is_lower_self_adjoint_matrix<Eigen::Replicate<MatrixType, 1, 1>>
    : std::bool_constant<lower_self_adjoint_matrix<MatrixType>> {};


  template<typename MatrixType>
  struct is_upper_self_adjoint_matrix<Eigen::Replicate<MatrixType, 1, 1>>
    : std::bool_constant<upper_self_adjoint_matrix<MatrixType>> {};


  template<typename MatrixType>
  struct is_lower_triangular_matrix<Eigen::Replicate<MatrixType, 1, 1>>
    : std::bool_constant<lower_triangular_matrix<MatrixType>> {};


  template<typename MatrixType>
  struct is_upper_triangular_matrix<Eigen::Replicate<MatrixType, 1, 1>>
    : std::bool_constant<upper_triangular_matrix<MatrixType>> {};


  // --------- //
  //  Reverse  //
  // --------- //

  namespace interface
  {
    template<typename MatrixType, int Direction>
    struct Dependencies<Eigen::Reverse<MatrixType, Direction>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename MatrixType::Nested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        using M = equivalent_self_contained_t<MatrixType>;
        if constexpr (not std::is_lvalue_reference_v<typename M::Nested>)
          return Eigen::Reverse<M, Direction> {make_self_contained(arg.nestedExpression())};
        else
          return make_native_matrix(std::forward<Arg>(arg));
      }
    };
  } // namespace interface


  /// The reverse of a constant matrix is constant.
#ifdef __cpp_concepts
  template<constant_matrix MatrixType, int Direction>
  struct constant_coefficient<Eigen::Reverse<MatrixType, Direction>>
#else
  template<typename MatrixType, int Direction>
  struct constant_coefficient<Eigen::Reverse<MatrixType, Direction>, std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient<std::decay_t<MatrixType>> {};


  /// The reverse of a constant matrix is constant-diagonal if it is square and zero.
#ifdef __cpp_concepts
  template<typename MatrixType, int Direction> requires (not constant_diagonal_matrix<MatrixType>) and
    square_matrix<MatrixType> and (constant_coefficient_v<MatrixType> == 0)
  struct constant_diagonal_coefficient<Eigen::Reverse<MatrixType, Direction>>
#else
  template<typename MatrixType, int Direction>
  struct constant_coefficient<Eigen::Reverse<MatrixType, Direction>, std::enable_if_t<
    (not constant_diagonal_matrix<MatrixType>) and square_matrix<MatrixType> and zero_matrix<MatrixType>>>
#endif
    : constant_coefficient<std::decay_t<MatrixType>> {};


  /// The double reverse of a constant-diagonal matrix, or any reverse of a zero or one-by-one matrix, is constant-diagonal.
#ifdef __cpp_concepts
  template<constant_diagonal_matrix MatrixType, int Direction> requires (Direction == Eigen::BothDirections) or
    one_by_one_matrix<MatrixType> or (constant_diagonal_coefficient_v<MatrixType> == 0)
  struct constant_diagonal_coefficient<Eigen::Reverse<MatrixType, Direction>>
#else
  template<typename MatrixType, int Direction>
  struct constant_diagonal_coefficient<Eigen::Reverse<MatrixType, Direction>, std::enable_if_t<
    constant_diagonal_matrix<MatrixType> and (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType> or
      constant_diagonal_coefficient<MatrixType>::value == 0)>>
#endif
    : constant_diagonal_coefficient<std::decay_t<MatrixType>> {};


  template<typename MatrixType, int Direction>
  struct is_diagonal_matrix<Eigen::Reverse<MatrixType, Direction>>
    : std::bool_constant<diagonal_matrix<MatrixType> and
    (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType>)> {};


  template<typename MatrixType, int Direction>
  struct is_lower_self_adjoint_matrix<Eigen::Reverse<MatrixType, Direction>>
    : std::bool_constant<upper_self_adjoint_matrix<MatrixType> and
    (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType>)> {};


  template<typename MatrixType, int Direction>
  struct is_upper_self_adjoint_matrix<Eigen::Reverse<MatrixType, Direction>>
    : std::bool_constant<lower_self_adjoint_matrix<MatrixType> and
    (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType>)> {};


  template<typename MatrixType, int Direction>
  struct is_lower_triangular_matrix<Eigen::Reverse<MatrixType, Direction>>
    : std::bool_constant<upper_triangular_matrix<MatrixType> and
    (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType>)> {};


  template<typename MatrixType, int Direction>
  struct is_upper_triangular_matrix<Eigen::Reverse<MatrixType, Direction>>
    : std::bool_constant<lower_triangular_matrix<MatrixType> and
    (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType>)> {};


  // -------- //
  //  Select  //
  // -------- //

  namespace interface
  {
    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct Dependencies<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
    {
    private:

      using T = Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>;
      using N = Eigen::Select<equivalent_self_contained_t<ConditionMatrixType>,
        equivalent_self_contained_t<ThenMatrixType>, equivalent_self_contained_t<ElseMatrixType>>;

    public:

      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename ConditionMatrixType::Nested, typename ThenMatrixType::Nested,
        typename ElseMatrixType::Nested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i < 3);
        if constexpr (i == 0)
          return std::forward<Arg>(arg).conditionMatrix();
        else if constexpr (i == 1)
          return std::forward<Arg>(arg).thenMatrix();
        else
          return std::forward<Arg>(arg).elseMatrix();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        // Do a partial evaluation as long as at least two arguments are already self-contained.
        if constexpr (
          ((self_contained<ConditionMatrixType> ? 1 : 0) + (self_contained<ThenMatrixType> ? 1 : 0) +
            (self_contained<ElseMatrixType> ? 1 : 0) >= 2) and
          not std::is_lvalue_reference_v<typename equivalent_self_contained_t<ConditionMatrixType>::Nested> and
          not std::is_lvalue_reference_v<typename equivalent_self_contained_t<ThenMatrixType>::Nested> and
          not std::is_lvalue_reference_v<typename equivalent_self_contained_t<ElseMatrixType>::Nested>)
        {
          return N {make_self_contained(arg.arg1()), make_self_contained(arg.arg2()), make_self_contained(arg.arg3()),
                    arg.functor()};
        }
        else
        {
          return make_native_matrix(std::forward<Arg>(arg));
        }
      }
    };
  }


  // --- constant_coefficient --- //

#ifdef __cpp_concepts
  template<constant_matrix ConditionMatrixType, constant_matrix ThenMatrixType, typename ElseMatrixType>
  requires constant_coefficient_v<ConditionMatrixType>
  struct constant_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct constant_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<constant_matrix<ConditionMatrixType> and constant_matrix<ThenMatrixType> and
      (constant_coefficient<ConditionMatrixType>::value)>>
#endif
    : constant_coefficient<std::decay_t<ThenMatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix ConditionMatrixType, typename ThenMatrixType, constant_matrix ElseMatrixType>
  requires (not constant_coefficient_v<ConditionMatrixType>)
  struct constant_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct constant_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<constant_matrix<ConditionMatrixType> and constant_matrix<ElseMatrixType> and
      (not constant_coefficient<ConditionMatrixType>::value)>>
#endif
    : constant_coefficient<std::decay_t<ElseMatrixType>> {};


#ifdef __cpp_concepts
  template<typename ConditionMatrixType, constant_matrix ThenMatrixType, constant_matrix ElseMatrixType>
  requires (not constant_matrix<ConditionMatrixType>) and
    (constant_coefficient_v<ThenMatrixType> == constant_coefficient_v<ElseMatrixType>)
  struct constant_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct constant_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<constant_matrix<ThenMatrixType> and constant_matrix<ElseMatrixType> and
      (not constant_matrix<ConditionMatrixType>) and
      (constant_coefficient<ThenMatrixType>::value == constant_coefficient<ElseMatrixType>::value)>>
#endif
    : constant_coefficient<std::decay_t<ThenMatrixType>> {};


  // --- constant_diagonal_coefficient --- //


  /// A constant selection is constant-diagonal if it is square and zero.
#ifdef __cpp_concepts
  template<constant_matrix ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  requires (not constant_diagonal_matrix<ThenMatrixType>) and (not constant_diagonal_matrix<ElseMatrixType>) and
    square_matrix<ConditionMatrixType> and
    (constant_coefficient_v<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>> == 0)
  struct constant_diagonal_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct constant_diagonal_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<constant_matrix<ConditionMatrixType> and (not constant_diagonal_matrix<ThenMatrixType>) and
    (not constant_diagonal_matrix<ElseMatrixType>) and square_matrix<ConditionMatrixType> and
    (constant_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>::value == 0)>>
#endif
    : constant_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix ConditionMatrixType, constant_diagonal_matrix ThenMatrixType, typename ElseMatrixType>
  requires constant_coefficient_v<ConditionMatrixType>
  struct constant_diagonal_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct constant_diagonal_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<constant_matrix<ConditionMatrixType> and constant_diagonal_matrix<ThenMatrixType> and
      constant_coefficient<ConditionMatrixType>::value>>
#endif
    : constant_diagonal_coefficient<std::decay_t<ThenMatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix ConditionMatrixType, typename ThenMatrixType, constant_diagonal_matrix ElseMatrixType>
  requires (not constant_coefficient_v<ConditionMatrixType>)
  struct constant_diagonal_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct constant_diagonal_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<(constant_matrix<ConditionMatrixType>) and constant_diagonal_matrix<ElseMatrixType> and
      (not constant_coefficient<ConditionMatrixType>::value)>>
#endif
    : constant_diagonal_coefficient<std::decay_t<ElseMatrixType>> {};


#ifdef __cpp_concepts
  template<typename ConditionMatrixType, constant_diagonal_matrix ThenMatrixType,
    constant_diagonal_matrix ElseMatrixType> requires (not constant_matrix<ConditionMatrixType>) and
    (constant_diagonal_coefficient_v<ThenMatrixType> == constant_diagonal_coefficient_v<ElseMatrixType>)
  struct constant_diagonal_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct constant_diagonal_coefficient<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<constant_diagonal_matrix<ThenMatrixType> and constant_diagonal_matrix<ElseMatrixType> and
      (not constant_matrix<ConditionMatrixType>) and
      (constant_diagonal_coefficient<ThenMatrixType>::value == constant_diagonal_coefficient<ElseMatrixType>::value)>>
#endif
    : constant_diagonal_coefficient<std::decay_t<ThenMatrixType>> {};


  // --- diagonal_matrix --- //


#ifdef __cpp_concepts
  template<constant_matrix ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct is_diagonal_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct is_diagonal_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<constant_matrix<ConditionMatrixType>>>
#endif
    : std::bool_constant<(diagonal_matrix<ThenMatrixType> and constant_coefficient_v<ConditionMatrixType>) or
        (diagonal_matrix<ElseMatrixType> and not constant_coefficient_v<ConditionMatrixType>)> {};


#ifdef __cpp_concepts
  template<constant_matrix ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct is_lower_self_adjoint_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct is_lower_self_adjoint_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<constant_matrix<ConditionMatrixType>>>
#endif
    : std::bool_constant<(lower_self_adjoint_matrix<ThenMatrixType> and constant_coefficient_v<ConditionMatrixType>) or
        (lower_self_adjoint_matrix<ElseMatrixType> and not constant_coefficient_v<ConditionMatrixType>)> {};


#ifdef __cpp_concepts
  template<self_adjoint_matrix ConditionMatrixType, self_adjoint_matrix ThenMatrixType,
    self_adjoint_matrix ElseMatrixType> requires (not constant_matrix<ConditionMatrixType>) and
    (lower_self_adjoint_matrix<ThenMatrixType> or lower_self_adjoint_matrix<ElseMatrixType>)
  struct is_lower_self_adjoint_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct is_lower_self_adjoint_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<self_adjoint_matrix<ConditionMatrixType> and self_adjoint_matrix<ThenMatrixType> and
      self_adjoint_matrix<ElseMatrixType> and (not constant_matrix<ConditionMatrixType>) and
      (lower_self_adjoint_matrix<ThenMatrixType> or lower_self_adjoint_matrix<ElseMatrixType>)>>
#endif
    : std::true_type {};


#ifdef __cpp_concepts
  template<constant_matrix ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct is_upper_self_adjoint_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct is_upper_self_adjoint_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<constant_matrix<ConditionMatrixType>>>
#endif
    : std::bool_constant<(upper_self_adjoint_matrix<ThenMatrixType> and constant_coefficient_v<ConditionMatrixType>) or
        (upper_self_adjoint_matrix<ElseMatrixType> and not constant_coefficient_v<ConditionMatrixType>)> {};


#ifdef __cpp_concepts
  template<self_adjoint_matrix ConditionMatrixType, upper_self_adjoint_matrix ThenMatrixType,
    upper_self_adjoint_matrix ElseMatrixType> requires (not constant_matrix<ConditionMatrixType>)
  struct is_upper_self_adjoint_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct is_upper_self_adjoint_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<self_adjoint_matrix<ConditionMatrixType> and upper_self_adjoint_matrix<ThenMatrixType> and
      upper_self_adjoint_matrix<ElseMatrixType> and (not constant_matrix<ConditionMatrixType>)>>
#endif
    : std::true_type {};


#ifdef __cpp_concepts
  template<constant_matrix ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct is_lower_triangular_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct is_lower_triangular_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<constant_matrix<ConditionMatrixType>>>
#endif
    : std::bool_constant<(lower_triangular_matrix<ThenMatrixType> and constant_coefficient_v<ConditionMatrixType>) or
        (lower_triangular_matrix<ElseMatrixType> and not constant_coefficient_v<ConditionMatrixType>)> {};


#ifdef __cpp_concepts
  template<constant_matrix ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct is_upper_triangular_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
  template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
  struct is_upper_triangular_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
    std::enable_if_t<constant_matrix<ConditionMatrixType>>>
#endif
    : std::bool_constant<(upper_triangular_matrix<ThenMatrixType> and constant_coefficient_v<ConditionMatrixType>) or
        (upper_triangular_matrix<ElseMatrixType> and not constant_coefficient_v<ConditionMatrixType>)> {};


  // ----------------- //
  //  SelfAdjointView  //
  // ----------------- //

  namespace interface
  {

    template<typename M, unsigned int UpLo>
    struct Dependencies<Eigen::SelfAdjointView<M, UpLo>>
    {
      static constexpr bool has_runtime_parameters = false;

      using type = std::tuple<typename Eigen::SelfAdjointView<M, UpLo>::MatrixTypeNested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        if constexpr (std::is_const_v<std::remove_reference_t<std::tuple_element_t<0, type>>>)
          return std::as_const(arg).nestedExpression();
        else
          return arg.nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        return make_self_contained(Eigen3::SelfAdjointMatrix {std::forward<Arg>(arg)});
      }
    };

  } // namespace interface


  /**
   * \internal
   * \brief Matrix traits for Eigen::SelfAdjointView.
   */
  template<typename M, unsigned int UpLo>
  struct MatrixTraits<Eigen::SelfAdjointView<M, UpLo>> : MatrixTraits<M>
  {
  private:

    using Scalar = typename Eigen::internal::traits<Eigen::SelfAdjointView<M, UpLo>>::Scalar;
    static constexpr auto rows = row_extent_of_v<M>;
    static constexpr auto columns = column_extent_of_v<M>;

    static constexpr TriangleType storage_triangle = UpLo & Eigen::Upper ? TriangleType::upper : TriangleType::lower;

  public:

    template<TriangleType storage_triangle = storage_triangle, std::size_t dim = rows, typename S = Scalar>
    using SelfAdjointMatrixFrom = typename MatrixTraits<M>::template SelfAdjointMatrixFrom<storage_triangle, dim, S>;


    template<TriangleType triangle_type = storage_triangle, std::size_t dim = rows, typename S = Scalar>
    using TriangularMatrixFrom = typename MatrixTraits<M>::template TriangularMatrixFrom<triangle_type, dim, S>;


#ifdef __cpp_concepts
    template<Eigen3::native_eigen_matrix Arg>
#else
    template<typename Arg, std::enable_if_t<Eigen3::native_eigen_matrix<Arg>, int> = 0>
#endif
    auto make(Arg& arg) noexcept
    {
      return Eigen::SelfAdjointView<std::remove_reference_t<Arg>, UpLo>(arg);
    }

  };


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, unsigned int UpLo>
  struct constant_coefficient<Eigen::SelfAdjointView<MatrixType, UpLo>>
#else
  template<typename MatrixType, unsigned int UpLo>
  struct constant_coefficient<Eigen::SelfAdjointView<MatrixType, UpLo>, std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient<std::decay_t<MatrixType>> {};


#ifdef __cpp_concepts
  template<zero_matrix MatrixType, unsigned int UpLo> requires (not constant_diagonal_matrix<MatrixType>)
  struct constant_diagonal_coefficient<Eigen::SelfAdjointView<MatrixType, UpLo>>
#else
  template<typename MatrixType, unsigned int UpLo>
  struct constant_diagonal_coefficient<Eigen::SelfAdjointView<MatrixType, UpLo>, std::enable_if_t<
    zero_matrix<MatrixType> and (not constant_diagonal_matrix<MatrixType>)>>
#endif
    : constant_coefficient<std::decay_t<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_diagonal_matrix MatrixType, unsigned int UpLo>
  struct constant_diagonal_coefficient<Eigen::SelfAdjointView<MatrixType, UpLo>>
#else
  template<typename MatrixType, unsigned int UpLo>
  struct constant_diagonal_coefficient<Eigen::SelfAdjointView<MatrixType, UpLo>, std::enable_if_t<
    constant_diagonal_matrix<MatrixType>>>
#endif
    : constant_diagonal_coefficient<std::decay_t<MatrixType>> {};


  template<typename MatrixType, unsigned int UpLo>
  struct is_diagonal_matrix<Eigen::SelfAdjointView<MatrixType, UpLo>>
    : std::bool_constant<diagonal_matrix<MatrixType>> {};


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct any_const_imag_part_is_zero
      : std::false_type {};

    template<typename T>
    struct any_const_imag_part_is_zero<T, std::enable_if_t<
      std::imag(constant_coefficient<std::decay_t<T>>::value) != 0>>
      : std::false_type {};

    template<typename T>
    struct any_const_imag_part_is_zero<T, std::enable_if_t<
      not constant_matrix<T> and std::imag(constant_diagonal_coefficient<std::decay_t<T>>::value) != 0>>
      : std::false_type {};
  };
#endif


#ifdef __cpp_concepts
  template<typename MatrixType, unsigned int UpLo> requires
    (not complex_number<typename Eigen::internal::traits<MatrixType>::Scalar>) or
    (std::imag(constant_coefficient_v<MatrixType>) == 0) or
    (std::imag(constant_diagonal_coefficient_v<MatrixType>) == 0)
  struct is_lower_self_adjoint_matrix<Eigen::SelfAdjointView<MatrixType, UpLo>>
#else
  template<typename MatrixType, unsigned int UpLo>
  struct is_lower_self_adjoint_matrix<Eigen::SelfAdjointView<MatrixType, UpLo>, std::enable_if_t<
    (not complex_number<typename Eigen::internal::traits<MatrixType>::Scalar>) or
    OpenKalman::detail::any_const_imag_part_is_zero<MatrixType>::value>>
#endif
    : std::bool_constant<(UpLo & Eigen::Lower) != 0> {};


#ifdef __cpp_concepts
  template<typename MatrixType, unsigned int UpLo> requires
    (not complex_number<typename Eigen::internal::traits<MatrixType>::Scalar>) or
    (std::imag(constant_coefficient_v<MatrixType>) == 0) or
    (std::imag(constant_diagonal_coefficient_v<MatrixType>) == 0)
  struct is_upper_self_adjoint_matrix<Eigen::SelfAdjointView<MatrixType, UpLo>>
#else
  template<typename MatrixType, unsigned int UpLo>
  struct is_upper_self_adjoint_matrix<Eigen::SelfAdjointView<MatrixType, UpLo>, std::enable_if_t<
    (not complex_number<typename Eigen::internal::traits<MatrixType>::Scalar>) or
    OpenKalman::detail::any_const_imag_part_is_zero<MatrixType>::value>>
#endif
    : std::bool_constant<(UpLo & Eigen::Upper) != 0> {};


  // ------- //
  //  Solve  //
  // ------- //

  namespace interface
  {
    template<typename Decomposition, typename RhsType>
    struct Dependencies<Eigen::Solve<Decomposition, RhsType>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<const Decomposition&, const RhsType&>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i < 2);
        if constexpr (i == 0)
          return std::forward<Arg>(arg).dec();
        else
          return std::forward<Arg>(arg).rhs();
      }

      // Eigen::Solve can never be self-contained.

    };


  // ----------- //
  //  Transpose  //
  // ----------- //

    template<typename MatrixType>
    struct Dependencies<Eigen::Transpose<MatrixType>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename Eigen::internal::ref_selector<MatrixType>::non_const_type>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        using M = equivalent_self_contained_t<MatrixType>;
        using N = Eigen::Transpose<M>;
        if constexpr (not std::is_lvalue_reference_v<typename Eigen::internal::ref_selector<M>::non_const_type>)
          return N {make_self_contained(arg.nestedExpression())};
        else
          return make_native_matrix(std::forward<Arg>(arg));
      }
    };

  } // namespace interface


#ifdef __cpp_concepts
  template<constant_matrix MatrixType>
  struct constant_coefficient<Eigen::Transpose<MatrixType>>
#else
  template<typename MatrixType>
  struct constant_coefficient<Eigen::Transpose<MatrixType>, std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    : constant_coefficient<std::decay_t<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType> requires square_matrix<MatrixType> and
    (constant_coefficient_v<MatrixType> == 0 or
      row_extent_of_v<MatrixType> == 1 or column_extent_of_v<MatrixType> == 1)
  struct constant_diagonal_coefficient<Eigen::Transpose<MatrixType>>
#else
  template<typename MatrixType>
  struct constant_diagonal_coefficient<Eigen::Transpose<MatrixType>, std::enable_if_t<
    constant_matrix<MatrixType> and square_matrix<MatrixType> and
    (zero_matrix<MatrixType> or row_extent_of<MatrixType>::value == 1 or column_extent_of<MatrixType>::value == 1)>>
#endif
    : constant_coefficient<std::decay_t<MatrixType>> {};


#ifdef __cpp_concepts
  template<constant_diagonal_matrix MatrixType>
  struct constant_diagonal_coefficient<Eigen::Transpose<MatrixType>>
#else
  template<typename MatrixType>
  struct constant_diagonal_coefficient<Eigen::Transpose<MatrixType>, std::enable_if_t<
    constant_diagonal_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<MatrixType>> {};


  template<typename MatrixType>
  struct is_diagonal_matrix<Eigen::Transpose<MatrixType>>
    : std::bool_constant<diagonal_matrix<MatrixType>> {};


  template<typename MatrixType>
  struct is_lower_self_adjoint_matrix<Eigen::Transpose<MatrixType>>
    : std::bool_constant<upper_self_adjoint_matrix<MatrixType>> {};


  template<typename MatrixType>
  struct is_upper_self_adjoint_matrix<Eigen::Transpose<MatrixType>>
    : std::bool_constant<lower_self_adjoint_matrix<MatrixType>> {};


  template<typename MatrixType>
  struct is_lower_triangular_matrix<Eigen::Transpose<MatrixType>>
    : std::bool_constant<upper_triangular_matrix<MatrixType>> {};


  template<typename MatrixType>
  struct is_upper_triangular_matrix<Eigen::Transpose<MatrixType>>
    : std::bool_constant<lower_triangular_matrix<MatrixType>> {};


  // ---------------- //
  //  TriangularView  //
  // ---------------- //

  namespace interface
  {

    template<typename M, unsigned int Mode>
    struct Dependencies<Eigen::TriangularView<M, Mode>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename Eigen::internal::traits<Eigen::TriangularView<M, Mode>>::MatrixTypeNested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        if constexpr (std::is_const_v<std::remove_reference_t<std::tuple_element_t<0, type>>>)
          return std::as_const(arg).nestedExpression();
        else
          return arg.nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        return make_self_contained(Eigen3::TriangularMatrix {std::forward<Arg>(arg)});
      }
    };

  } // namespace interface


  /**
   * \internal
   * \brief Matrix traits for Eigen::TriangularView.
   */
  template<typename M, unsigned int Mode>
  struct MatrixTraits<Eigen::TriangularView<M, Mode>> : MatrixTraits<M>
  {
  private:

    using Scalar = typename Eigen::internal::traits<Eigen::TriangularView<M, Mode>>::Scalar;
    static constexpr auto rows = row_extent_of_v<M>;
    static constexpr auto columns = column_extent_of_v<M>;

    static constexpr TriangleType triangle_type = Mode & Eigen::Upper ? TriangleType::upper : TriangleType::lower;

  public:

    template<TriangleType storage_triangle = triangle_type, std::size_t dim = rows, typename S = Scalar>
    using SelfAdjointMatrixFrom = typename MatrixTraits<M>::template SelfAdjointMatrixFrom<storage_triangle, dim, S>;


    template<TriangleType triangle_type = triangle_type, std::size_t dim = rows, typename S = Scalar>
    using TriangularMatrixFrom = typename MatrixTraits<M>::template TriangularMatrixFrom<triangle_type, dim, S>;


#ifdef __cpp_concepts
    template<Eigen3::native_eigen_matrix Arg>
#else
    template<typename Arg, std::enable_if_t<Eigen3::native_eigen_matrix<Arg>, int> = 0>
#endif
    auto make(Arg& arg) noexcept
    {
      return Eigen::TriangularView<std::remove_reference_t<Arg>, Mode>(arg);
    }

  };


#ifdef __cpp_concepts
  template<diagonal_matrix MatrixType, unsigned int Mode> requires ((Mode & Eigen::UnitDiag) != 0)
  struct constant_coefficient<Eigen::TriangularView<MatrixType, Mode>>
#else
  template<typename MatrixType, unsigned int Mode>
  struct constant_coefficient<Eigen::TriangularView<MatrixType, Mode>, std::enable_if_t<
    diagonal_matrix<MatrixType> and (Mode & Eigen::UnitDiag) != 0>>
#endif
    : constant_coefficient_type<1, short> {};


#ifdef __cpp_concepts
  template<diagonal_matrix MatrixType, unsigned int Mode> requires ((Mode & Eigen::ZeroDiag) != 0)
  struct constant_coefficient<Eigen::TriangularView<MatrixType, Mode>>
#else
  template<typename MatrixType, unsigned int Mode>
  struct constant_coefficient<Eigen::TriangularView<MatrixType, Mode>, std::enable_if_t<
    diagonal_matrix<MatrixType> and (Mode & Eigen::ZeroDiag) != 0>>
#endif
    : constant_coefficient_type<0, short> {};


#ifdef __cpp_concepts
  template<constant_matrix MatrixType, unsigned int Mode> requires
    ((Mode & Eigen::UnitDiag) == 0) and ((Mode & Eigen::ZeroDiag) == 0) and
    (constant_coefficient_v<MatrixType> == 0 or one_by_one_matrix<MatrixType>)
  struct constant_coefficient<Eigen::TriangularView<MatrixType, Mode>>
#else
  template<typename MatrixType, unsigned int Mode>
  struct constant_coefficient<Eigen::TriangularView<MatrixType, Mode>, std::enable_if_t<constant_matrix<MatrixType> and
    ((Mode & Eigen::UnitDiag) == 0) and ((Mode & Eigen::ZeroDiag) == 0) and
    (zero_matrix<MatrixType> or one_by_one_matrix<MatrixType>)>>
#endif
    : constant_coefficient<std::decay_t<MatrixType>> {};


#ifdef __cpp_concepts
  template<triangular_matrix MatrixType, unsigned int Mode> requires
    (not constant_diagonal_matrix<MatrixType>) and ((Mode & Eigen::UnitDiag) != 0) and
    (((Mode & Eigen::Upper) != 0 and lower_triangular_matrix<MatrixType>) or
      ((Mode & Eigen::Lower) != 0 and upper_triangular_matrix<MatrixType>))
  struct constant_diagonal_coefficient<Eigen::TriangularView<MatrixType, Mode>>
#else
  template<typename MatrixType, unsigned int Mode>
  struct constant_diagonal_coefficient<Eigen::TriangularView<MatrixType, Mode>, std::enable_if_t<
    triangular_matrix<MatrixType> and (not constant_diagonal_matrix<MatrixType>) and (Mode & Eigen::UnitDiag) != 0 and
    (((Mode & Eigen::Upper) != 0 and lower_triangular_matrix<MatrixType>) or
      ((Mode & Eigen::Lower) != 0 and upper_triangular_matrix<MatrixType>))>>
#endif
    : constant_coefficient_type<1, short> {};


#ifdef __cpp_concepts
  template<triangular_matrix MatrixType, unsigned int Mode> requires
    (not constant_diagonal_matrix<MatrixType>) and ((Mode & Eigen::ZeroDiag) != 0) and
    (((Mode & Eigen::Upper) != 0 and lower_triangular_matrix<MatrixType>) or
      ((Mode & Eigen::Lower) != 0 and upper_triangular_matrix<MatrixType>))
  struct constant_diagonal_coefficient<Eigen::TriangularView<MatrixType, Mode>>
#else
  template<typename MatrixType, unsigned int Mode>
  struct constant_diagonal_coefficient<Eigen::TriangularView<MatrixType, Mode>, std::enable_if_t<
    triangular_matrix<MatrixType> and (not constant_diagonal_matrix<MatrixType>) and (Mode & Eigen::ZeroDiag) != 0 and
    (((Mode & Eigen::Upper) != 0 and lower_triangular_matrix<MatrixType>) or
      ((Mode & Eigen::Lower) != 0 and upper_triangular_matrix<MatrixType>))>>
#endif
    : constant_coefficient_type<0, short> {};


#ifdef __cpp_concepts
  template<typename MatrixType, unsigned int Mode> requires (not constant_diagonal_matrix<MatrixType>) and
    (constant_coefficient_v<MatrixType> == 0) and ((Mode & Eigen::UnitDiag) == 0) and ((Mode & Eigen::ZeroDiag) == 0)
  struct constant_diagonal_coefficient<Eigen::TriangularView<MatrixType, Mode>>
#else
  template<typename MatrixType, unsigned int Mode>
  struct constant_diagonal_coefficient<Eigen::TriangularView<MatrixType, Mode>, std::enable_if_t<
    (not constant_diagonal_matrix<MatrixType>) and zero_matrix<MatrixType> and
    ((Mode & Eigen::UnitDiag) == 0) and ((Mode & Eigen::ZeroDiag) == 0)>>
#endif
    : constant_coefficient_type<0, short> {};


#ifdef __cpp_concepts
  template<constant_diagonal_matrix MatrixType, unsigned int Mode>
  struct constant_diagonal_coefficient<Eigen::TriangularView<MatrixType, Mode>>
#else
  template<typename MatrixType, unsigned int Mode>
  struct constant_diagonal_coefficient<Eigen::TriangularView<MatrixType, Mode>, std::enable_if_t<
    constant_diagonal_matrix<MatrixType>>>
#endif
    : constant_coefficient_type<(Mode & Eigen::UnitDiag) != 0 ? 1 : ((Mode & Eigen::ZeroDiag) != 0 ? 0 :
    constant_diagonal_coefficient_v<MatrixType>), constant_diagonal_coefficient_t<MatrixType>> {};


  template<typename MatrixType, unsigned int Mode>
  struct is_diagonal_matrix<Eigen::TriangularView<MatrixType, Mode>>
    : std::bool_constant<((Mode & Eigen::Lower) != 0 and upper_triangular_matrix<MatrixType>) or
      ((Mode & Eigen::Upper) != 0 and lower_triangular_matrix<MatrixType>)> {};


  template<typename MatrixType, unsigned int Mode>
  struct is_lower_triangular_matrix<Eigen::TriangularView<MatrixType, Mode>>
    : std::bool_constant<lower_triangular_matrix<MatrixType> or (Mode & Eigen::Lower) != 0> {};


  template<typename MatrixType, unsigned int Mode>
  struct is_upper_triangular_matrix<Eigen::TriangularView<MatrixType, Mode>>
    : std::bool_constant<upper_triangular_matrix<MatrixType> or (Mode & Eigen::Upper) != 0> {};


  // ------------- //
  //  VectorBlock  //
  // ------------- //

  namespace interface
  {
    template<typename VectorType, int Size>
    struct Dependencies<Eigen::VectorBlock<VectorType, Size>>
    {
      static constexpr bool has_runtime_parameters = true;
      using type = std::tuple<typename Eigen::internal::ref_selector<VectorType>::non_const_type>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      // Eigen::VectorBlock should always be converted to Matrix

    };
  } // namespace interface


  /// A segment taken from a constant vector is constant.
#ifdef __cpp_concepts
  template<constant_matrix VectorType, int Size>
  struct constant_coefficient<Eigen::VectorBlock<VectorType, Size>>
#else
    template<typename VectorType, int Size>
    struct constant_coefficient<Eigen::VectorBlock<VectorType, Size>, std::enable_if_t<constant_matrix<VectorType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<VectorType>> {};


  /// A segment taken from a constant vector is constant-diagonal if it is one-by-one.
#ifdef __cpp_concepts
  template<constant_matrix VectorType, int Size> requires (Size == 1 or one_by_one_matrix<VectorType>)
  struct constant_diagonal_coefficient<Eigen::VectorBlock<VectorType, Size>>
#else
  template<typename VectorType, int Size>
  struct constant_diagonal_coefficient<Eigen::VectorBlock<VectorType, Size>, std::enable_if_t<
    constant_matrix<VectorType> and (Size == 1 or one_by_one_matrix<VectorType>)>>
#endif
    : constant_coefficient_type<constant_coefficient_v<VectorType>> {};


  // -------------- //
  //  VectorWiseOp  //
  // -------------- //

  namespace interface
  {

    template<typename ExpressionType, int Direction>
    struct Dependencies<Eigen::VectorwiseOp<ExpressionType, Direction>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename ExpressionType::ExpressionTypeNested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg)._expression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        using N = Eigen::VectorwiseOp<ExpressionType, Direction>;
        if constexpr (not std::is_lvalue_reference_v<typename N::ExpressionTypeNested>)
          return N {make_self_contained(arg._expression())};
        else
          return make_native_matrix(std::forward<Arg>(arg));
      }
    };

  } // namespace interface


  /// A vectorwise operation on a constant vector is constant.
#ifdef __cpp_concepts
  template<constant_matrix ExpressionType, int Direction>
  struct constant_coefficient<Eigen::VectorwiseOp<ExpressionType, Direction>>
#else
  template<typename ExpressionType, int Direction>
  struct constant_coefficient<Eigen::VectorwiseOp<ExpressionType, Direction>,
    std::enable_if_t<constant_matrix<ExpressionType>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<ExpressionType>> {};


} // namespace OpenKalman::internal

#endif //OPENKALMAN_EIGEN3_TRAITS_HPP
