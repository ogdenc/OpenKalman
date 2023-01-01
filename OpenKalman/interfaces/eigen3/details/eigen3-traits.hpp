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
  namespace EGI = Eigen::internal;


#ifdef __cpp_concepts
  template<native_eigen_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<native_eigen_matrix<Arg>, int> = 0>
#endif
  explicit SelfAdjointMatrix(Arg&&) -> SelfAdjointMatrix<passable_t<Arg>, TriangleType::lower>;


#ifdef __cpp_concepts
  template<TriangleType t = TriangleType::lower, native_eigen_matrix M>
#else
  template<TriangleType t = TriangleType::lower, typename M, std::enable_if_t<native_eigen_matrix<M>, int> = 0>
#endif
  auto
  make_EigenSelfAdjointMatrix(M&& m)
  {
    return SelfAdjointMatrix<passable_t<M>, t> {std::forward<M>(m)};
  }


#ifdef __cpp_concepts
  template<native_eigen_matrix M>
#else
  template<typename M, std::enable_if_t<native_eigen_matrix<M>, int> = 0>
#endif
  explicit TriangularMatrix(M&&) -> TriangularMatrix<passable_t<M>, TriangleType::lower>;


#ifdef __cpp_concepts
  template<TriangleType t = TriangleType::lower, native_eigen_matrix M>
#else
 template<TriangleType t = TriangleType::lower, typename M, std::enable_if_t<native_eigen_matrix<M>, int> = 0>
#endif
  auto make_EigenTriangularMatrix(M&& m)
  {
    return TriangularMatrix<passable_t<M>, t> (std::forward<M>(m));
  }


  namespace interface
  {

  // ---------------------- //
  //  native_eigen_general  //
  // ---------------------- //

#ifdef __cpp_concepts
    template<native_eigen_general T>
    struct IndexibleObjectTraits<T>
#else
    template<typename T>
    struct IndexibleObjectTraits<T, std::enable_if_t<native_eigen_general<T>>>
#endif
    {
      static constexpr std::size_t max_indices = 2;
      using scalar_type = typename std::decay_t<T>::Scalar;
    };


#ifdef __cpp_concepts
    template<native_eigen_general T>
    struct IndexTraits<T, 0>
#else
    template<typename T>
    struct IndexTraits<T, 0, std::enable_if_t<native_eigen_general<T>>>
#endif
    {
    private:

      static constexpr auto e_dim =
        (std::decay_t<T>::RowsAtCompileTime == Eigen::Dynamic) and
          (eigen_SelfAdjointView<T> or eigen_TriangularView<T>) ? std::decay_t<T>::ColsAtCompileTime :
        std::decay_t<T>::RowsAtCompileTime;

    public:

      static constexpr std::size_t dimension = e_dim == Eigen::Dynamic ? dynamic_size : static_cast<std::size_t>(e_dim);

      template<typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        if constexpr (dynamic_rows<Arg>)
          return static_cast<std::size_t>(arg.rows());
        else
          return dimension;
      }
    };


#ifdef __cpp_concepts
    template<native_eigen_general T>
    struct IndexTraits<T, 1>
#else
    template<typename T>
    struct IndexTraits<T, 1, std::enable_if_t<native_eigen_general<T>>>
#endif
    {
    private:

      static constexpr auto e_dim =
        (std::decay_t<T>::ColsAtCompileTime == Eigen::Dynamic) and (
          eigen_SelfAdjointView<T> or eigen_TriangularView<T>) ? std::decay_t<T>::RowsAtCompileTime :
        std::decay_t<T>::ColsAtCompileTime;

    public:

      static constexpr std::size_t dimension = e_dim == Eigen::Dynamic ? dynamic_size : static_cast<std::size_t>(e_dim);

      template<typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        if constexpr (dynamic_columns<Arg>)
          return static_cast<std::size_t>(arg.cols());
        else
          return dimension;
      }
    };


#ifdef __cpp_concepts
    template<native_eigen_general T, typename Scalar>
    struct EquivalentDenseWritableMatrix<T, Scalar>
#else
    template<typename T, typename Scalar>
    struct EquivalentDenseWritableMatrix<T, Scalar, std::enable_if_t<native_eigen_general<T>>>
#endif
    {
    private:

      template<Eigen::Index...Args>
      using dense_type = std::conditional_t<native_eigen_array<T>,
        Eigen::Array<Scalar, Args...>, Eigen::Matrix<Scalar, Args...>>;

      template<std::size_t...Args>
      using type = dense_type<(Args == dynamic_size ? Eigen::Dynamic : static_cast<Eigen::Index>(Args))...>;

#  ifdef __cpp_concepts
      template<typename ArgType>
#  else
      template<typename ArgType, typename = void>
#  endif
      struct traits_and_evaluator_defined : std::false_type {};


#  ifdef __cpp_concepts
       template<typename ArgType> requires
         requires { typename Eigen::internal::traits<ArgType>; typename Eigen::internal::evaluator<ArgType>; }
       struct traits_and_evaluator_defined<ArgType> : std::true_type {};
#  else
       template<typename ArgType>
       struct traits_and_evaluator_defined<ArgType, std::enable_if_t<
         std::is_void<std::void_t<Eigen::internal::traits<ArgType>>>::value and
         std::is_void<std::void_t<Eigen::internal::evaluator<ArgType>>>::value>> : std::true_type {};
#  endif

    public:

      static constexpr bool is_writable = (native_eigen_matrix<T> or native_eigen_array<T>) and
        static_cast<bool>(Eigen::internal::traits<std::decay_t<T>>::Flags & (Eigen::LvalueBit | Eigen::DirectAccessBit));


      template<typename...D>
      static auto make_default(D&&...d)
      {
        using M = type<dimension_size_of_v<D>...>;

        if constexpr (((dimension_size_of_v<D> == dynamic_size) or ...))
          return M(static_cast<Eigen::Index>(get_dimension_size_of(d))...);
        else
          return M {};
      }


      template<typename Arg>
      static decltype(auto) convert(Arg&& arg)
      {
        using M = type<index_dimension_of_v<Arg, 0>, index_dimension_of_v<Arg, 1>>;

        if constexpr (eigen_DiagonalWrapper<Arg>)
        {
          // Note: Arg's nested matrix might not be a column vector.
          return M {to_diagonal(Conversions<Arg>::diagonal_of(std::forward<Arg>(arg)))};
        }
        else if constexpr (std::is_base_of_v<Eigen::PlainObjectBase<std::decay_t<Arg>>, std::decay_t<Arg>> and
          not std::is_const_v<std::remove_reference_t<Arg>>)
        {
          return std::forward<Arg>(arg);
        }
        else if constexpr (std::is_constructible_v<M, Arg&&>)
        {
          return M {std::forward<Arg>(arg)};
        }
        else
        {
          auto r = get_index_dimension_of<0>(arg);
          auto c = get_index_dimension_of<1>(arg);
          auto m = make_default(r, c);
          for (int i = 0; i < r ; ++i) for (int j = 0; j < c ; ++j)
          {
            set_element(m, get_element(std::forward<Arg>(arg), i, j), i, j);
          }
          return m;
        }
      }


      template<typename Arg>
      static decltype(auto) to_native_matrix(Arg&& arg)
      {
        if constexpr (native_eigen_matrix<Arg>)
          return std::forward<Arg>(arg);
        else if constexpr (native_eigen_array<Arg>)
          return std::forward<Arg>(arg).matrix();
        else if constexpr (native_eigen_general<Arg>)
          return convert(std::forward<Arg>(arg));
        else
        {
          static_assert(traits_and_evaluator_defined<Arg>::value, "To convert to a native Eigen matrix, the interface "
            "must define a trait and an evaluator for the argument");
          return EigenWrapper<std::decay_t<Arg>> {std::forward<Arg>(arg)};
        }
      }


      template<typename Arg>
      static decltype(auto) to_native_matrix(const EigenWrapper<Arg>& arg) { return arg; }


      template<typename Arg>
      static decltype(auto) to_native_matrix(EigenWrapper<Arg>&& arg) { return std::move(arg); }

    };


#ifdef __cpp_concepts
    template<native_eigen_general T, typename Scalar>
    struct SingleConstantMatrixTraits<T, Scalar>
#else
    template<typename T, typename Scalar>
    struct SingleConstantMatrixTraits<T, Scalar, std::enable_if_t<native_eigen_general<T>>>
#endif
    {
      template<typename...D>
      static constexpr auto make_zero_matrix(D&&...d)
      {
        using N = dense_writable_matrix_t<T, std::decay_t<D>...>;
        return ZeroAdapter<N> {std::forward<D>(d)...};
      }

      template<auto constant, typename...D>
      static constexpr auto make_constant_matrix(D&&...d)
      {
        using N = dense_writable_matrix_t<T, std::decay_t<D>...>;
        return ConstantAdapter<N, constant> {std::forward<D>(d)...};
      }
    };


#ifdef __cpp_concepts
    template<native_eigen_general T, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<T, Scalar>
#else
    template<typename T, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<T, Scalar, std::enable_if_t<native_eigen_general<T>>>
#endif
    {
      template<typename D>
      static constexpr auto make_identity_matrix(D&& d)
      {
        if constexpr (dimension_size_of_v<D> == dynamic_size)
        {
          return to_diagonal(make_constant_matrix_like<T, 1, Scalar>(std::forward<D>(d), Dimensions<1>{}));
        }
        else
        {
          constexpr Eigen::Index n {dimension_size_of_v<D>};
          return eigen_matrix_t<Scalar, n, n>::Identity();
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
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
      }
    };


#ifdef __cpp_concepts
    template<constant_matrix XprType>
    struct SingleConstant<Eigen::ArrayWrapper<XprType>>
#else
    template<typename XprType>
    struct SingleConstant<Eigen::ArrayWrapper<XprType>, std::enable_if_t<constant_matrix<XprType>>>
#endif
      : SingleConstant<std::decay_t<XprType>> {};


#ifdef __cpp_concepts
    template<constant_diagonal_matrix XprType>
    struct SingleConstantDiagonal<Eigen::ArrayWrapper<XprType>>
#else
    template<typename XprType>
    struct SingleConstantDiagonal<Eigen::ArrayWrapper<XprType>, std::enable_if_t<constant_diagonal_matrix<XprType>>>
#endif
      : SingleConstantDiagonal<std::decay_t<XprType>> {};


    template<typename XprType>
    struct DiagonalTraits<Eigen::ArrayWrapper<XprType>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<XprType>;
    };


#ifdef __cpp_concepts
    template<triangular_matrix XprType>
    struct TriangularTraits<Eigen::ArrayWrapper<XprType>>
#else
    template<typename XprType>
    struct TriangularTraits<Eigen::ArrayWrapper<XprType>, std::enable_if_t<triangular_matrix<XprType>>>
#endif
    {
      static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;
      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<hermitian_matrix XprType>
    struct HermitianTraits<Eigen::ArrayWrapper<XprType>>
#else
    template<typename XprType>
    struct HermitianTraits<Eigen::ArrayWrapper<XprType>, std::enable_if_t<hermitian_matrix<XprType>>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


    // ------- //
    //  Block  //
    // ------- //

    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    struct Dependencies<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
    {
      static constexpr bool has_runtime_parameters = true;
      using type = std::tuple<typename EGI::ref_selector<XprType>::non_const_type>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      // Eigen::Block should always be converted to Matrix

    };


    // A block taken from a constant matrix is constant.
#ifdef __cpp_concepts
    template<constant_matrix XprType, int BlockRows, int BlockCols, bool InnerPanel>
    struct SingleConstant<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
#else
    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    struct SingleConstant<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>,
      std::enable_if_t<constant_matrix<XprType>>>
#endif
      : SingleConstant<std::decay_t<XprType>> {};


    // A block taken from a constant matrix is constant-diagonal if it is square and either zero or one-by-one.
#ifdef __cpp_concepts
    template<constant_matrix XprType, int BlockRows, int BlockCols, bool InnerPanel>
    requires (BlockRows == BlockCols) and (zero_matrix<XprType> or BlockRows == 1 or BlockCols == 1)
    struct SingleConstantDiagonal<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
#else
    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    struct SingleConstantDiagonal<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>, std::enable_if_t<
      constant_matrix<XprType> and (BlockRows == BlockCols) and
      (zero_matrix<XprType> or BlockRows == 1 or BlockCols == 1)>>
#endif
      : SingleConstant<std::decay_t<XprType>> {};


  // --------------- //
  //  CwiseBinaryOp  //
  // --------------- //

    template<typename BinaryOp, typename LhsType, typename RhsType>
    struct Dependencies<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
    {
    private:

      using T = Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>;

    public:

      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename T::LhsNested, typename T::RhsNested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i <= 2);
        if constexpr (i == 0)
          return std::forward<Arg>(arg).lhs();
        else if constexpr (i == 2)
          return std::forward<Arg>(arg).rhs();
        else
          return std::forward<Arg>(arg).functor();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        using N = Eigen::CwiseBinaryOp<BinaryOp, equivalent_self_contained_t<LhsType>,
          equivalent_self_contained_t<RhsType>>;
        // Do a partial evaluation as long as at least one argument is already self-contained.
        if constexpr ((self_contained<LhsType> or self_contained<RhsType>) and
          not std::is_lvalue_reference_v<typename N::LhsNested> and
          not std::is_lvalue_reference_v<typename N::RhsNested>)
        {
          return N {make_self_contained(arg.lhs()), make_self_contained(arg.rhs()), arg.functor()};
        }
        else
        {
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
        }
      }
    };

    // --- SingleConstant --- //

    // The sum of two constant matrices is also constant.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>,
      std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    {
      static constexpr auto value = constant_coefficient_v<Arg1> + constant_coefficient_v<Arg2>;
    };


    /// The sum of two constant-diagonal matrices is zero if the matrices cancel out.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
    requires (not constant_matrix<Arg1>) and (not constant_matrix<Arg2>) and
      (are_within_tolerance(constant_diagonal_coefficient_v<Arg1> + constant_diagonal_coefficient_v<Arg2>, 0))
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>, std::enable_if_t<
      constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2> and
      (not constant_matrix<Arg1>) and (not constant_matrix<Arg2>) and
      are_within_tolerance(constant_diagonal_coefficient<Arg1>::value + constant_diagonal_coefficient<Arg2>::value, 0)>>
#endif
    {
      static constexpr auto value = 0;
    };


    /// The difference between two constant matrices is also constant.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>,
      std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    {
      static constexpr auto value = constant_coefficient_v<Arg1> - constant_coefficient_v<Arg2>;
    };


    /// The difference between two constant-diagonal matrices is zero if the matrices are equal.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
    requires (not constant_matrix<Arg1>) and (not constant_matrix<Arg2>) and
      (are_within_tolerance(constant_diagonal_coefficient_v<Arg1>, constant_diagonal_coefficient_v<Arg2>))
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>,
      std::enable_if_t<constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2> and
      (not constant_matrix<Arg1>) and (not constant_matrix<Arg2>) and
      are_within_tolerance(constant_diagonal_coefficient<Arg1>::value, constant_diagonal_coefficient<Arg2>::value)>>
#endif
    {
      static constexpr auto value = 0;
    };


    /// The coefficient-wise product of two constant arrays is also constant.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>,
      std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    {
      static constexpr auto value = constant_coefficient_v<Arg1> * constant_coefficient_v<Arg2>;
    };


    /// The coefficient-wise product that includes a zero matrix is zero.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2> requires
      (not constant_matrix<Arg1> and zero_matrix<Arg2>) or (zero_matrix<Arg1> and not constant_matrix<Arg2>)
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>,
      std::enable_if_t<(not constant_matrix<Arg1> and zero_matrix<Arg2>) or
        (zero_matrix<Arg1> and not constant_matrix<Arg2>)>>
#endif
    {
      static constexpr auto value = 0;
    };


    /// The coefficient-wise conjugate product of two constant arrays is also constant.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_conj_product_op<Scalar1, Scalar2>,
      Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<
      EGI::scalar_conj_product_op<Scalar1, Scalar2>, Arg1, Arg2>,
      std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    {
      static constexpr auto value = constant_coefficient_v<Arg1> * constant_coefficient_v<Arg2>;
    };


    /// The coefficient-wise conjugate product that includes a zero matrix is zero.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2> requires
      (not constant_matrix<Arg1> and zero_matrix<Arg2>) or (zero_matrix<Arg1> and not constant_matrix<Arg2>)
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_conj_product_op<Scalar1, Scalar2>,
      Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_conj_product_op<Scalar1, Scalar2>,
      Arg1, Arg2>, std::enable_if_t<
        (not constant_matrix<Arg1> and zero_matrix<Arg2>) or (zero_matrix<Arg1> and not constant_matrix<Arg2>)>>
#endif
    {
      static constexpr auto value = 0;
    };


    /// The coefficient-wise quotient of two constant arrays is also constant.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
    requires (not zero_matrix<Arg2>)
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_quotient_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_quotient_op<Scalar1, Scalar2>, Arg1, Arg2>,
      std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2> and (not zero_matrix<Arg2>)>>
#endif
    {
# if __cpp_nontype_template_args >= 201911L
      static constexpr auto value = constant_coefficient_v<Arg1> / constant_coefficient_v<Arg2>;
# else
      static constexpr auto value = [] {
        using Scalar = scalar_type_of_t<Eigen::CwiseBinaryOp<EGI::scalar_quotient_op<Scalar1, Scalar2>, Arg1, Arg2>>;
        if constexpr (std::is_integral_v<decltype(constant_coefficient_v<Arg1>)> and
          std::is_integral_v<decltype(constant_coefficient_v<Arg2>)> and
          constant_coefficient_v<Arg1> % constant_coefficient_v<Arg2> != 0)
        {
          return static_cast<Scalar>(constant_coefficient_v<Arg1>) / static_cast<Scalar>(constant_coefficient_v<Arg2>);
        }
        else
        {
          return constant_coefficient_v<Arg1> / constant_coefficient_v<Arg2>;
        }
      }();
# endif
    };


    /// The coefficient-wise quotient of a zero matrix and another matrix is zero.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, zero_matrix Arg1, typename Arg2> requires (not constant_matrix<Arg2>)
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_quotient_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_quotient_op<Scalar1, Scalar2>, Arg1, Arg2>,
      std::enable_if_t<(not constant_matrix<Arg2>) and zero_matrix<Arg1>>>
#endif
    {
      static constexpr auto value = 0;
    };


    /// The coefficient-wise min of two constant arrays is also constant.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_min_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_min_op<Scalar1, Scalar2>, Arg1, Arg2>,
      std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    {
      static constexpr auto value = constant_coefficient_v<Arg1> <= constant_coefficient_v<Arg2> ? constant_coefficient_v<Arg1> :
          constant_coefficient_v<Arg2>;
    };


    /// The coefficient-wise max of two constant arrays is also constant.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_max_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_max_op<Scalar1, Scalar2>, Arg1, Arg2>,
      std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    {
      static constexpr auto value = constant_coefficient_v<Arg1> >= constant_coefficient_v<Arg2> ? constant_coefficient_v<Arg1> :
          constant_coefficient_v<Arg2>;
    };


    /// The coefficient-wise hypotenuse of two constant arrays is also constant.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, constant_matrix Arg1, constant_matrix Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_hypot_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_hypot_op<Scalar1, Scalar2>, Arg1, Arg2>,
      std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    {
    private:
      using Scalar = scalar_type_of_t<Eigen::CwiseBinaryOp<EGI::scalar_hypot_op<Scalar1, Scalar2>, Arg1, Arg2>>;
    public:
      static constexpr auto value = OpenKalman::internal::constexpr_sqrt(static_cast<Scalar>(
        constant_coefficient_v<Arg1> * constant_coefficient_v<Arg1> +
        constant_coefficient_v<Arg2> * constant_coefficient_v<Arg2>));
    };


    /// The coefficient-wise power of two constant arrays is also constant. \todo update this with constexpr floating power
#ifdef __cpp_concepts
    template<typename Scalar, typename Exponent, constant_matrix Arg1, constant_matrix Arg2> requires
      std::is_integral_v<decltype(constant_coefficient_v<Arg2>)>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_pow_op<Scalar, Exponent>, Arg1, Arg2>>
#else
    template<typename Scalar, typename Exponent, typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_pow_op<Scalar, Exponent>, Arg1, Arg2>,
      std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2> and
      std::is_integral_v<decltype(constant_coefficient_v<Arg2>)>>>
#endif
    {
      static constexpr auto value =
        OpenKalman::internal::constexpr_pow(constant_coefficient_v<Arg1>, constant_coefficient_v<Arg2>);
    };


    /// The coefficient-wise AND of two constant arrays is also constant.
#ifdef __cpp_concepts
    template<constant_matrix Arg1, constant_matrix Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_boolean_and_op, Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_boolean_and_op, Arg1, Arg2>,
      std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    {
      static constexpr auto value = static_cast<bool>(constant_coefficient_v<Arg1>) and static_cast<bool>(constant_coefficient_v<Arg2>);
    };


    /// The coefficient-wise OR of two constant arrays is also constant.
#ifdef __cpp_concepts
    template<constant_matrix Arg1, constant_matrix Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_boolean_or_op, Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_boolean_or_op, Arg1, Arg2>,
      std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    {
      static constexpr auto value = static_cast<bool>(constant_coefficient_v<Arg1>) or static_cast<bool>(constant_coefficient_v<Arg2>);
    };


    /// The coefficient-wise XOR of two constant arrays is also constant.
#ifdef __cpp_concepts
    template<constant_matrix Arg1, constant_matrix Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_boolean_xor_op, Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::CwiseBinaryOp<EGI::scalar_boolean_xor_op, Arg1, Arg2>,
      std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2>>>
#endif
    {
      static constexpr auto value = static_cast<bool>(constant_coefficient_v<Arg1>) xor static_cast<bool>(constant_coefficient_v<Arg2>);
    };


    // --- constant_diagonal_coefficient --- //

    /// The result of a constant binary operation is constant-diagonal if it is square and zero.
#ifdef __cpp_concepts
    template<typename Op, typename Arg1, typename Arg2>
    requires (not constant_diagonal_matrix<Arg1> or not constant_diagonal_matrix<Arg2>) and
      (square_matrix<Arg1> or square_matrix<Arg2>) and
      zero_matrix<Eigen::CwiseBinaryOp<Op, Arg1, Arg2>>
    struct SingleConstantDiagonal<Eigen::CwiseBinaryOp<Op, Arg1, Arg2>>
#else
    template<typename Op, typename Arg1, typename Arg2>
    struct SingleConstantDiagonal<Eigen::CwiseBinaryOp<Op, Arg1, Arg2>, std::enable_if_t<
      (not constant_diagonal_matrix<Arg1> or not constant_diagonal_matrix<Arg2>) and
      (square_matrix<Arg1> or square_matrix<Arg2>) and
      zero_matrix<Eigen::CwiseBinaryOp<Op, Arg1, Arg2>>>>
#endif
      : SingleConstant<Eigen::CwiseBinaryOp<Op, Arg1, Arg2>> {};


    /// The sum of two constant-diagonal matrices is also constant-diagonal.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
    struct SingleConstantDiagonal<Eigen::CwiseBinaryOp<EGI::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct SingleConstantDiagonal<Eigen::CwiseBinaryOp<EGI::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>, std::enable_if_t<
      constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    {
      static constexpr auto value = constant_diagonal_coefficient_v<Arg1> + constant_diagonal_coefficient_v<Arg2>;
    };


    /// The difference between two constant-diagonal matrices is also constant-diagonal.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
    struct SingleConstantDiagonal<Eigen::CwiseBinaryOp<EGI::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct SingleConstantDiagonal<
      Eigen::CwiseBinaryOp<EGI::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>, std::enable_if_t<
        constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    {
      static constexpr auto value = constant_diagonal_coefficient_v<Arg1> - constant_diagonal_coefficient_v<Arg2>;
    };


    /// The coefficient-wise product of two constant-diagonal arrays is also constant-diagonal.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
    struct SingleConstantDiagonal<Eigen::CwiseBinaryOp<EGI::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct SingleConstantDiagonal<Eigen::CwiseBinaryOp<EGI::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>,
      std::enable_if_t<constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    {
      static constexpr auto value = constant_diagonal_coefficient_v<Arg1> * constant_diagonal_coefficient_v<Arg2>;
    };


    /// The coefficient-wise conjugate product of two constant-diagonal arrays is also constant-diagonal.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
    struct SingleConstantDiagonal<Eigen::CwiseBinaryOp<EGI::scalar_conj_product_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct SingleConstantDiagonal<Eigen::CwiseBinaryOp<
      EGI::scalar_conj_product_op<Scalar1, Scalar2>, Arg1, Arg2>,
      std::enable_if_t<constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    {
      static constexpr auto value = constant_diagonal_coefficient_v<Arg1> * constant_diagonal_coefficient_v<Arg2>;
    };


    /// The coefficient-wise quotient of two constant-diagonal arrays is also constant if it is a one-by-one matrix.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
    requires (one_by_one_matrix<Arg1> or one_by_one_matrix<Arg2>) and (not zero_matrix<Arg2>)
    struct SingleConstantDiagonal<Eigen::CwiseBinaryOp<EGI::scalar_quotient_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct SingleConstantDiagonal<Eigen::CwiseBinaryOp<EGI::scalar_quotient_op<Scalar1, Scalar2>,
      Arg1, Arg2>, std::enable_if_t<constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2> and
      (one_by_one_matrix<Arg1> or one_by_one_matrix<Arg2>) and (not zero_matrix<Arg2>)>>
#endif
    {
# if __cpp_nontype_template_args >= 201911L
      static constexpr auto value = constant_diagonal_coefficient_v<Arg1> / constant_diagonal_coefficient_v<Arg2>;
# else
      static constexpr auto value = [] {
        using Scalar = scalar_type_of_t<Eigen::CwiseBinaryOp<EGI::scalar_quotient_op<Scalar1, Scalar2>, Arg1, Arg2>>;
        if constexpr (std::is_integral_v<decltype(constant_diagonal_coefficient_v<Arg1>)> and
          std::is_integral_v<decltype(constant_diagonal_coefficient_v<Arg2>)> and
        constant_diagonal_coefficient_v<Arg1> % constant_diagonal_coefficient_v<Arg2> != 0)
        {
          return static_cast<Scalar>(constant_diagonal_coefficient_v<Arg1>) /
            static_cast<Scalar>(constant_diagonal_coefficient_v<Arg2>);
        }
        else
        {
          return constant_diagonal_coefficient_v<Arg1> / constant_diagonal_coefficient_v<Arg2>;
        }
      }();
# endif
    };


    /// The coefficient-wise min of two constant-diagonal arrays is also constant-diagonal.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
    struct SingleConstantDiagonal<Eigen::CwiseBinaryOp<EGI::scalar_min_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct SingleConstantDiagonal<Eigen::CwiseBinaryOp<EGI::scalar_min_op<Scalar1, Scalar2>, Arg1, Arg2>,
      std::enable_if_t<constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    {
      static constexpr auto value = constant_diagonal_coefficient_v<Arg1> <= constant_diagonal_coefficient_v<Arg2> ?
          constant_diagonal_coefficient_v<Arg1> : constant_diagonal_coefficient_v<Arg2>;
    };


    /// The coefficient-wise max of two constant-diagonal arrays is also constant-diagonal.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
    struct SingleConstantDiagonal<Eigen::CwiseBinaryOp<EGI::scalar_max_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct SingleConstantDiagonal<Eigen::CwiseBinaryOp<EGI::scalar_max_op<Scalar1, Scalar2>, Arg1, Arg2>,
      std::enable_if_t<constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    {
      static constexpr auto value = constant_diagonal_coefficient_v<Arg1> >= constant_diagonal_coefficient_v<Arg2> ?
          constant_diagonal_coefficient_v<Arg1> : constant_diagonal_coefficient_v<Arg2>;
    };


    /// The coefficient-wise hypotenuse of two constant-diagonal arrays is also constant-diagonal.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
    struct SingleConstantDiagonal<Eigen::CwiseBinaryOp<EGI::scalar_hypot_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct SingleConstantDiagonal<Eigen::CwiseBinaryOp<EGI::scalar_hypot_op<Scalar1, Scalar2>, Arg1, Arg2>,
      std::enable_if_t<constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    {
    private:
      using Scalar = scalar_type_of_t<Eigen::CwiseBinaryOp<EGI::scalar_hypot_op<Scalar1, Scalar2>, Arg1, Arg2>>;
    public:
      static constexpr auto value = constexpr_sqrt(static_cast<Scalar>(
        constant_diagonal_coefficient_v<Arg1> * constant_diagonal_coefficient_v<Arg1> +
        constant_diagonal_coefficient_v<Arg2> * constant_diagonal_coefficient_v<Arg2>));
    };


    // --- DiagonalTraits --- //

    /// The sum of two diagonal matrices is also diagonal.
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct DiagonalTraits<Eigen::CwiseBinaryOp<EGI::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<Arg1> and diagonal_matrix<Arg2>;
    };


    /// The difference between two diagonal matrices is also diagonal.
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct DiagonalTraits<Eigen::CwiseBinaryOp<EGI::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<Arg1> and diagonal_matrix<Arg2>;
    };


    /// A diagonal times another array, or the product of upper and lower triangular arrays ( in either order) is diagonal.
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct DiagonalTraits<Eigen::CwiseBinaryOp<EGI::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<Arg1> or diagonal_matrix<Arg2> or
        (lower_triangular_matrix<Arg1> and upper_triangular_matrix<Arg2>) or
        (upper_triangular_matrix<Arg1> and lower_triangular_matrix<Arg2>);
    };


    // --- TriangularTraits --- //

    /// The sum of two matching triangular matrices may be triangular.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, triangular_matrix Arg1, triangular_matrix Arg2>
    struct TriangularTraits<Eigen::CwiseBinaryOp<EGI::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct TriangularTraits<Eigen::CwiseBinaryOp<EGI::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>, std::enable_if_t<
      triangular_matrix<Arg1> and triangular_matrix<Arg2>>>
#endif
    {
      static constexpr TriangleType triangle_type = triangle_type_of_v<Arg1, Arg2>;
      static constexpr bool is_triangular_adapter = false;
    };


    /// The difference between two matching triangular matrices may be triangular.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, triangular_matrix Arg1, triangular_matrix Arg2>
    struct TriangularTraits<Eigen::CwiseBinaryOp<EGI::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct TriangularTraits<Eigen::CwiseBinaryOp<EGI::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>, std::enable_if_t<
      triangular_matrix<Arg1> and triangular_matrix<Arg2>>>
#endif
    {
      static constexpr TriangleType triangle_type = triangle_type_of_v<Arg1, Arg2>;
      static constexpr bool is_triangular_adapter = false;
    };


    /// The scalar product of two triangular matrices may be triangular.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, triangular_matrix Arg1, triangular_matrix Arg2>
    struct TriangularTraits<Eigen::CwiseBinaryOp<EGI::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct TriangularTraits<Eigen::CwiseBinaryOp<EGI::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>, std::enable_if_t<
      triangular_matrix<Arg1> and triangular_matrix<Arg2>>>
#endif
    {
      static constexpr TriangleType triangle_type = triangle_type_of_v<Arg1, Arg2>;
      static constexpr bool is_triangular_adapter = false;
    };


    // --- HermitianTraits --- //

    // The sum of two hermitian matrices is hermitian.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, hermitian_matrix Arg1, hermitian_matrix Arg2>
    struct HermitianTraits<Eigen::CwiseBinaryOp<EGI::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct HermitianTraits<Eigen::CwiseBinaryOp<EGI::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>, std::enable_if_t<
      hermitian_matrix<Arg1> and hermitian_matrix<Arg2>>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


    // The difference between two hermitian matrices is hermitian.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, hermitian_matrix Arg1, hermitian_matrix Arg2>
    struct HermitianTraits<Eigen::CwiseBinaryOp<EGI::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct HermitianTraits<Eigen::CwiseBinaryOp<EGI::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>, std::enable_if_t<
      hermitian_matrix<Arg1> and hermitian_matrix<Arg2>>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


    /// The scalar product of two hermitian matrices is hermitian if the two argument matrices are real.
#ifdef __cpp_concepts
    template<typename Scalar1, typename Scalar2, hermitian_matrix Arg1, hermitian_matrix Arg2> requires
      (not complex_number<Scalar1>) and (not complex_number<Scalar2>)
    struct HermitianTraits<Eigen::CwiseBinaryOp<EGI::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>>
#else
    template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
    struct HermitianTraits<Eigen::CwiseBinaryOp<EGI::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>, std::enable_if_t<
      hermitian_matrix<Arg1> and hermitian_matrix<Arg2> and
      not complex_number<Scalar1> and not complex_number<Scalar2>>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


  // ---------------- //
  //  CwiseNullaryOp  //
  // ---------------- //

    namespace detail
    {
      template<typename NullaryOp, typename PlainObjectType>
      struct CwiseNullaryOpDependenciesBase
      {
        static constexpr bool has_runtime_parameters =
          Eigen::CwiseNullaryOp<NullaryOp, PlainObjectType>::RowsAtCompileTime == Eigen::Dynamic or
          Eigen::CwiseNullaryOp<NullaryOp, PlainObjectType>::ColsAtCompileTime == Eigen::Dynamic;

        using type = std::tuple<>;

        template<std::size_t i, typename Arg>
        static decltype(auto) get_nested_matrix(Arg&& arg)
        {
          static_assert(i == 0);
          return std::forward<Arg>(arg).functor();
        }
      };
    }

    template<typename NullaryOp, typename PlainObjectType>
    struct Dependencies<Eigen::CwiseNullaryOp<NullaryOp, PlainObjectType>>
      : detail::CwiseNullaryOpDependenciesBase<NullaryOp, PlainObjectType> {};


    template<typename Scalar, typename PlainObjectType>
    struct Dependencies<Eigen::CwiseNullaryOp<EGI::scalar_constant_op<Scalar>, PlainObjectType>>
      : detail::CwiseNullaryOpDependenciesBase<EGI::scalar_constant_op<Scalar>, PlainObjectType>
    {
      static constexpr bool has_runtime_parameters = true;
    };


    template<typename PlainObjectType, typename...Args>
    struct Dependencies<Eigen::CwiseNullaryOp<EGI::linspaced_op<Args...>, PlainObjectType>>
      : detail::CwiseNullaryOpDependenciesBase<EGI::linspaced_op<Args...>, PlainObjectType>
    {
      static constexpr bool has_runtime_parameters = true;
    };


    // \brief An Eigen nullary operation is constant if it is identity and one-by-one.
#ifdef __cpp_concepts
    template<typename Scalar, one_by_one_matrix Arg>
    struct SingleConstant<Eigen::CwiseNullaryOp<EGI::scalar_identity_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstant<Eigen::CwiseNullaryOp<EGI::scalar_identity_op<Scalar>, Arg>,
        std::enable_if_t<one_by_one_matrix<Arg>>>
#endif
    {
      static constexpr auto value = 1;
    };


    // \brief An Eigen nullary operation is constant-diagonal if it is identity and square.
#ifdef __cpp_concepts
    template<typename Scalar, square_matrix Arg>
    struct SingleConstantDiagonal<Eigen::CwiseNullaryOp<EGI::scalar_identity_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstantDiagonal<Eigen::CwiseNullaryOp<EGI::scalar_identity_op<Scalar>, Arg>,
        std::enable_if_t<square_matrix<Arg>>>
#endif
    {
      static constexpr auto value = 1;
    };


    /// A constant square matrix is hermitian if it is not complex.
    template<typename Scalar, typename PlainObjectType>
    struct HermitianTraits<Eigen::CwiseNullaryOp<EGI::scalar_constant_op<Scalar>, PlainObjectType>>
    {
      static constexpr bool is_hermitian = square_matrix<PlainObjectType> and (not complex_number<Scalar>);
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


    // ---------------- //
    //  CwiseTernaryOp  //
    // ---------------- //

    template<typename TernaryOp, typename Arg1, typename Arg2, typename Arg3>
    struct Dependencies<Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>
    {
    private:

      using T = Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>;

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
        using N = Eigen::CwiseTernaryOp<TernaryOp,
          equivalent_self_contained_t<Arg1>, equivalent_self_contained_t<Arg2>, equivalent_self_contained_t<Arg3>>;
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
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
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

    public:

      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename T::XprTypeNested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        if constexpr (i == 0)
          return std::forward<Arg>(arg).nestedExpression();
        else
          return std::forward<Arg>(arg).functor();
        static_assert(i <= 1);
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        using N = Eigen::CwiseUnaryOp<UnaryOp, equivalent_self_contained_t<XprType>>;
        if constexpr (not std::is_lvalue_reference_v<typename N::XprTypeNested>)
          return N {make_self_contained(arg.nestedExpression()), arg.functor()};
        else
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
      }
    };


    template<typename BinaryOp, typename XprType>
    struct Dependencies<Eigen::CwiseUnaryOp<EGI::bind1st_op<BinaryOp>, XprType>>
    {
      static constexpr bool has_runtime_parameters = true;
      using type =
        std::tuple<typename Eigen::CwiseUnaryOp<EGI::bind1st_op<BinaryOp>, XprType>::XprTypeNested>;
    };


    template<typename BinaryOp, typename XprType>
    struct Dependencies<Eigen::CwiseUnaryOp<EGI::bind2nd_op<BinaryOp>, XprType>>
    {
      static constexpr bool has_runtime_parameters = true;
      using type =
        std::tuple<typename Eigen::CwiseUnaryOp<EGI::bind2nd_op<BinaryOp>, XprType>::XprTypeNested>;
    };

    // --- SingleConstant --- //

    /// The negation of a constant matrix is constant.
#ifdef __cpp_concepts
    template<typename Scalar, constant_matrix Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_opposite_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_opposite_op<Scalar>, Arg>,
      std::enable_if_t<constant_matrix<Arg>>>
#endif
    {
      static constexpr auto value = - constant_coefficient_v<Arg>;
    };


    /// The conjugate of a constant matrix is constant.
#ifdef __cpp_concepts
    template<typename Scalar, constant_matrix Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_conjugate_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_conjugate_op<Scalar>, Arg>,
      std::enable_if_t<constant_matrix<Arg>>>
#endif
    {
      static constexpr auto value = conjugate(constant_coefficient_v<Arg>);
    };


    /// The real part of a constant matrix is constant.
#ifdef __cpp_concepts
    template<typename Scalar, constant_matrix Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_real_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_real_op<Scalar>, Arg>,
      std::enable_if_t<constant_matrix<Arg>>>
#endif
    {
      static constexpr auto value = real_projection(constant_coefficient_v<Arg>);
    };


    /// The imaginary part of a constant matrix is constant.
#ifdef __cpp_concepts
    template<typename Scalar, constant_matrix Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_imag_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_imag_op<Scalar>, Arg>,
      std::enable_if_t<constant_matrix<Arg>>>
#endif
    {
      static constexpr auto value = imaginary_part(constant_coefficient_v<Arg>);
    };


    /// The absolute value of a constant array is constant.
#ifdef __cpp_concepts
    template<typename Scalar, constant_matrix Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_abs_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_abs_op<Scalar>, Arg>,
      std::enable_if_t<constant_matrix<Arg>>>
#endif
    {
      static constexpr auto value = constant_coefficient_v<Arg> >= 0 ?
        constant_coefficient_v<Arg> : -constant_coefficient_v<Arg>;
    };


    /// The squared absolute value of a constant array is constant.
#ifdef __cpp_concepts
    template<typename Scalar, constant_matrix Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_abs2_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_abs2_op<Scalar>, Arg>,
      std::enable_if_t<constant_matrix<Arg>>>
#endif
    {
      static constexpr auto value = complex_number<Scalar> ?
        real_projection(constant_coefficient_v<Arg>) * real_projection(constant_coefficient_v<Arg>) +
          imaginary_part(constant_coefficient_v<Arg>) * imaginary_part(constant_coefficient_v<Arg>):
        constant_coefficient_v<Arg> * constant_coefficient_v<Arg>;
    };


    /// The square root a constant array is constant.
#ifdef __cpp_concepts
    template<typename Scalar, constant_matrix Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_sqrt_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_sqrt_op<Scalar>, Arg>,
      std::enable_if_t<constant_matrix<Arg>>>
#endif
    {
      static constexpr auto value = OpenKalman::internal::constexpr_sqrt(
        static_cast<Scalar>(constant_coefficient_v<Arg>));
    };


    /// The inverse of a constant array is constant.
#ifdef __cpp_concepts
    template<typename Scalar, constant_matrix Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_inverse_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_inverse_op<Scalar>, Arg>,
      std::enable_if_t<constant_matrix<Arg>>>
#endif
    {
      static constexpr auto value = static_cast<Scalar>(1) / constant_coefficient_v<Arg>;
    };


    /// The square of a constant array is constant.
#ifdef __cpp_concepts
    template<typename Scalar, constant_matrix Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_square_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_square_op<Scalar>, Arg>,
      std::enable_if_t<constant_matrix<Arg>>>
#endif
    {
      static constexpr auto value = constant_coefficient_v<Arg> * constant_coefficient_v<Arg>;
    };



    /// The cube of a constant array is constant.
#ifdef __cpp_concepts
    template<typename Scalar, constant_matrix Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_cube_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_cube_op<Scalar>, Arg>,
      std::enable_if_t<constant_matrix<Arg>>>
#endif
    {
      static constexpr auto value =
        constant_coefficient_v<Arg> * constant_coefficient_v<Arg> * constant_coefficient_v<Arg>;
    };



    /// The logical not of a constant array is constant.
#ifdef __cpp_concepts
    template<typename Scalar, constant_matrix Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_boolean_not_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstant<Eigen::CwiseUnaryOp<EGI::scalar_boolean_not_op<Scalar>, Arg>,
      std::enable_if_t<constant_matrix<Arg>>>
#endif
    {
      static constexpr auto value = not static_cast<bool>(constant_coefficient_v<Arg>);
    };


    // --- constant_diagonal_coefficient --- //

    /// The result of a unary operation is constant-diagonal if it is square and zero.
#ifdef __cpp_concepts
    template<typename Op, typename Arg> requires (not constant_diagonal_matrix<Arg>) and
      square_matrix<Arg> and zero_matrix<Eigen::CwiseUnaryOp<Op, Arg>>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<Op, Arg>>
#else
    template<typename Op, typename Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<Op, Arg>, std::enable_if_t<
      (not constant_diagonal_matrix<Arg>) and square_matrix<Arg> and
      zero_matrix<Eigen::CwiseUnaryOp<Op, Arg>>>>
#endif
      : SingleConstant<Eigen::CwiseUnaryOp<Op, Arg>> {};


    /// The negation of a constant-diagonal matrix is also constant-diagonal.
#ifdef __cpp_concepts
    template<typename Scalar, constant_diagonal_matrix Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_opposite_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_opposite_op<Scalar>, Arg>,
      std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    {
      static constexpr auto value = -constant_diagonal_coefficient_v<Arg>;
    };


    /// The conjugate of a constant-diagonal matrix is constant-diagonal.
#ifdef __cpp_concepts
    template<typename Scalar, constant_diagonal_matrix Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_conjugate_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_conjugate_op<Scalar>, Arg>,
      std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    {
      static constexpr auto value = conjugate(constant_diagonal_coefficient_v<Arg>);
    };


    /// The real part of a constant-diagonal matrix is also constant-diagonal.
#ifdef __cpp_concepts
    template<typename Scalar, constant_diagonal_matrix Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_real_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_real_op<Scalar>, Arg>,
      std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    {
      static constexpr auto value = real_projection(constant_diagonal_coefficient_v<Arg>);
    };


    /// The imaginary part of a constant-diagonal matrix is also constant-diagonal.
#ifdef __cpp_concepts
    template<typename Scalar, constant_diagonal_matrix Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_imag_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_imag_op<Scalar>, Arg>,
      std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    {
      static constexpr auto value = imaginary_part(constant_diagonal_coefficient_v<Arg>);
    };


    /// The coefficient-wise absolute value of a constant-diagonal array is also constant-diagonal.
#ifdef __cpp_concepts
    template<typename Scalar, constant_diagonal_matrix Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_abs_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_abs_op<Scalar>, Arg>,
      std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    {
      static constexpr auto value = constant_diagonal_coefficient_v<Arg> >= 0 ?
        constant_diagonal_coefficient_v<Arg> : -constant_diagonal_coefficient_v<Arg>;
    };


    /// The coefficient-wise squared absolute value of a constant-diagonal array is also constant-diagonal.
#ifdef __cpp_concepts
    template<typename Scalar, constant_diagonal_matrix Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_abs2_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_abs2_op<Scalar>, Arg>,
      std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    {
      static constexpr auto value = constant_diagonal_coefficient_v<Arg> * constant_diagonal_coefficient_v<Arg>;
    };


    /// The coefficient-wise square root a constant-diagonal array is also constant-diagonal.
#ifdef __cpp_concepts
    template<typename Scalar, constant_diagonal_matrix Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_sqrt_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_sqrt_op<Scalar>, Arg>,
      std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    {
      static constexpr auto value = OpenKalman::internal::constexpr_sqrt(
        static_cast<Scalar>(constant_diagonal_coefficient_v<Arg>));
    };


    /// The coefficient-wise inverse of a constant-diagonal array is also constant-diagonal.
#ifdef __cpp_concepts
    template<typename Scalar, constant_diagonal_matrix Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_inverse_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_inverse_op<Scalar>, Arg>,
      std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    {
      static constexpr auto value = static_cast<Scalar>(1) / constant_diagonal_coefficient_v<Arg>;
    };


    /// The coefficient-wise square of a constant-diagonal array is also constant-diagonal.
#ifdef __cpp_concepts
    template<typename Scalar, constant_diagonal_matrix Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_square_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_square_op<Scalar>, Arg>,
      std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    {
      static constexpr auto value = constant_diagonal_coefficient_v<Arg> * constant_diagonal_coefficient_v<Arg>;
    };


    /// The coefficient-wise cube of a constant-diagonal array is also constant-diagonal.
#ifdef __cpp_concepts
    template<typename Scalar, constant_diagonal_matrix Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_cube_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_cube_op<Scalar>, Arg>,
      std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    {
      static constexpr auto value = constant_diagonal_coefficient_v<Arg> * constant_diagonal_coefficient_v<Arg> *
          constant_diagonal_coefficient_v<Arg>;
    };


    /// The logical not of a constant-diagonal array is constant-diagonal if it is one-by-one.
#ifdef __cpp_concepts
    template<typename Scalar, constant_diagonal_matrix Arg> requires one_by_one_matrix<Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_boolean_not_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryOp<EGI::scalar_boolean_not_op<Scalar>, Arg>,
      std::enable_if_t<constant_diagonal_matrix<Arg> and one_by_one_matrix<Arg>>>
#endif
    {
      static constexpr auto value = not static_cast<bool>(constant_diagonal_coefficient_v<Arg>);
    };


    template<typename Scalar, typename Arg>
    struct DiagonalTraits<Eigen::CwiseUnaryOp<EGI::scalar_real_op<Scalar>, Arg>>
      : std::bool_constant<diagonal_matrix<Arg>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<Arg>;
    };


    template<typename Scalar, typename Arg>
    struct DiagonalTraits<Eigen::CwiseUnaryOp<EGI::scalar_imag_op<Scalar>, Arg>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<Arg>;
    };


    /// The negation of a diagonal matrix is also diagonal.
    template<typename Scalar, typename Arg>
    struct DiagonalTraits<Eigen::CwiseUnaryOp<EGI::scalar_opposite_op<Scalar>, Arg>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<Arg>;
    };


    /// The conjugate of a diagonal matrix is also diagonal.
    template<typename Scalar, typename Arg>
    struct DiagonalTraits<Eigen::CwiseUnaryOp<EGI::scalar_conjugate_op<Scalar>, Arg>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<Arg>;
    };


    namespace detail
    {
      template<typename T>
      struct triangular_preserving_unary_op : std::false_type {};

      template<typename Scalar>
      struct triangular_preserving_unary_op<EGI::scalar_opposite_op<Scalar>> : std::true_type {};

      template<typename Scalar>
      struct triangular_preserving_unary_op<EGI::scalar_conjugate_op<Scalar>> : std::true_type {};

      template<typename Scalar>
      struct triangular_preserving_unary_op<EGI::scalar_real_op<Scalar>> : std::true_type {};

      template<typename Scalar>
      struct triangular_preserving_unary_op<EGI::scalar_real_ref_op<Scalar>> : std::true_type {};

      template<typename Scalar>
      struct triangular_preserving_unary_op<EGI::scalar_imag_op<Scalar>> : std::true_type {};

      template<typename Scalar>
      struct triangular_preserving_unary_op<EGI::scalar_imag_ref_op<Scalar>> : std::true_type {};

      template<typename Scalar>
      struct triangular_preserving_unary_op<EGI::scalar_abs_op<Scalar>> : std::true_type {};

      template<typename Scalar>
      struct triangular_preserving_unary_op<EGI::scalar_abs2_op<Scalar>> : std::true_type {};

      template<typename Scalar>
      struct triangular_preserving_unary_op<EGI::scalar_sqrt_op<Scalar>> : std::true_type {};

      template<typename Scalar>
      struct triangular_preserving_unary_op<EGI::scalar_inverse_op<Scalar>> : std::true_type {};

      template<typename Scalar>
      struct triangular_preserving_unary_op<EGI::scalar_square_op<Scalar>> : std::true_type {};

      template<typename Scalar>
      struct triangular_preserving_unary_op<EGI::scalar_cube_op<Scalar>> : std::true_type {};

      template<typename Scalar>
      struct triangular_preserving_unary_op<EGI::scalar_boolean_not_op<Scalar>> : std::false_type {};


      template<typename T>
      struct hermitian_preserving_unary_op : std::false_type {};

      template<typename Scalar>
      struct hermitian_preserving_unary_op<EGI::scalar_opposite_op<Scalar>> : std::true_type {};

      template<typename Scalar>
      struct hermitian_preserving_unary_op<EGI::scalar_conjugate_op<Scalar>> : std::true_type {};

      template<typename Scalar>
      struct hermitian_preserving_unary_op<EGI::scalar_real_op<Scalar>> : std::true_type {};

      template<typename Scalar>
      struct hermitian_preserving_unary_op<EGI::scalar_real_ref_op<Scalar>> : std::true_type {};

      template<typename Scalar>
      struct hermitian_preserving_unary_op<EGI::scalar_imag_op<Scalar>>
        : std::bool_constant<not complex_number<Scalar>> {};

      template<typename Scalar>
      struct hermitian_preserving_unary_op<EGI::scalar_imag_ref_op<Scalar>>
        : std::bool_constant<not complex_number<Scalar>> {};

      template<typename Scalar>
      struct hermitian_preserving_unary_op<EGI::scalar_abs_op<Scalar>> : std::true_type {};

      template<typename Scalar>
      struct hermitian_preserving_unary_op<EGI::scalar_abs2_op<Scalar>> : std::true_type {};

      template<typename Scalar>
      struct hermitian_preserving_unary_op<EGI::scalar_sqrt_op<Scalar>>
        : std::bool_constant<not complex_number<Scalar>> {};

      template<typename Scalar>
      struct hermitian_preserving_unary_op<EGI::scalar_inverse_op<Scalar>> : std::true_type {};

      template<typename Scalar>
      struct hermitian_preserving_unary_op<EGI::scalar_square_op<Scalar>> : std::true_type {};

      template<typename Scalar>
      struct hermitian_preserving_unary_op<EGI::scalar_cube_op<Scalar>> : std::true_type {};

      template<typename Scalar>
      struct hermitian_preserving_unary_op<EGI::scalar_boolean_not_op<Scalar>> : std::true_type {};
    }


#ifdef __cpp_concepts
    template<typename UnaryOp, triangular_matrix Arg> requires one_by_one_matrix<Arg> or
      detail::triangular_preserving_unary_op<UnaryOp>::value
    struct TriangularTraits<Eigen::CwiseUnaryOp<UnaryOp, Arg>>
#else
    template<typename UnaryOp, typename Arg>
    struct TriangularTraits<Eigen::CwiseUnaryOp<UnaryOp, Arg>, std::enable_if_t<
      triangular_matrix<Arg> and (one_by_one_matrix<Arg> or detail::triangular_preserving_unary_op<UnaryOp>::value)>>
#endif
    {
      static constexpr TriangleType triangle_type = triangle_type_of_v<Arg>;
      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<typename UnaryOp, hermitian_matrix Arg> requires one_by_one_matrix<Arg> or
      (not diagonal_matrix<Arg> and detail::hermitian_preserving_unary_op<UnaryOp>::value) or
      (diagonal_matrix<Arg> and detail::triangular_preserving_unary_op<UnaryOp>::value)
    struct HermitianTraits<Eigen::CwiseUnaryOp<UnaryOp, Arg>>
#else
    template<typename UnaryOp, typename Arg>
    struct HermitianTraits<Eigen::CwiseUnaryOp<UnaryOp, Arg>, std::enable_if_t<hermitian_matrix<Arg> and
      (one_by_one_matrix<Arg> or
      (not diagonal_matrix<Arg> and detail::hermitian_preserving_unary_op<UnaryOp>::value) or
      (diagonal_matrix<Arg> and detail::triangular_preserving_unary_op<UnaryOp>::value))>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


    // ---------------- //
    //  CwiseUnaryView  //
    // ---------------- //

    template<typename ViewOp, typename MatrixType>
    struct Dependencies<Eigen::CwiseUnaryView<ViewOp, MatrixType>>
    {
    private:

      using T = Eigen::CwiseUnaryView<ViewOp, MatrixType>;

    public:

      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename T::MatrixTypeNested, ViewOp>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        if constexpr (i == 0)
          return std::forward<Arg>(arg).nestedExpression();
        else
          return std::forward<Arg>(arg).functor();
        static_assert(i <= 1);
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        using N = Eigen::CwiseUnaryView<ViewOp, equivalent_self_contained_t<MatrixType>>;
        if constexpr (not std::is_lvalue_reference_v<typename N::MatrixTypeNested>)
          return N {make_self_contained(arg.nestedExpression()), arg.functor()};
        else
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
      }
    };


    /// The real part of a constant matrix is constant.
#ifdef __cpp_concepts
    template<typename Scalar, constant_matrix Arg>
    struct SingleConstant<Eigen::CwiseUnaryView<EGI::scalar_real_ref_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstant<Eigen::CwiseUnaryView<EGI::scalar_real_ref_op<Scalar>, Arg>,
      std::enable_if_t<constant_matrix<Arg>>>
#endif
    {
      static constexpr auto value = real_projection(constant_coefficient_v<Arg>);
    };



    /// The imaginary part of a constant matrix is constant.
#ifdef __cpp_concepts
    template<typename Scalar, constant_matrix Arg>
    struct SingleConstant<Eigen::CwiseUnaryView<EGI::scalar_imag_ref_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstant<Eigen::CwiseUnaryView<EGI::scalar_imag_ref_op<Scalar>, Arg>,
      std::enable_if_t<constant_matrix<Arg>>>
#endif
    {
      static constexpr auto value = imaginary_part(constant_coefficient_v<Arg>);
    };


    /// A square, zero constant CwiseUnaryView is also constant-diagonal.
#ifdef __cpp_concepts
    template<typename Op, square_matrix Arg> requires (not constant_diagonal_matrix<Arg>) and
      zero_matrix<Eigen::CwiseUnaryView<Op, Arg>>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryView<Op, Arg>>
#else
    template<typename Op, typename Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryView<Op, Arg>, std::enable_if_t<square_matrix<Arg> and
      (not constant_diagonal_matrix<Arg>) and zero_matrix<Eigen::CwiseUnaryView<Op, Arg>>>>
#endif
      : SingleConstant<Eigen::CwiseUnaryView<Op, Arg>> {};


    /// The real part of a constant-diagonal matrix is also constant-diagonal.
#ifdef __cpp_concepts
    template<typename Scalar, constant_diagonal_matrix Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryView<EGI::scalar_real_ref_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryView<EGI::scalar_real_ref_op<Scalar>, Arg>,
      std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    {
      static constexpr auto value = real_projection(constant_diagonal_coefficient_v<Arg>);
    };


    /// The imaginary part of a constant-diagonal matrix is also constant-diagonal.
#ifdef __cpp_concepts
    template<typename Scalar, constant_diagonal_matrix Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryView<EGI::scalar_imag_ref_op<Scalar>, Arg>>
#else
    template<typename Scalar, typename Arg>
    struct SingleConstantDiagonal<Eigen::CwiseUnaryView<EGI::scalar_imag_ref_op<Scalar>, Arg>,
      std::enable_if_t<constant_diagonal_matrix<Arg>>>
#endif
    {
      static constexpr auto value = imaginary_part(constant_diagonal_coefficient_v<Arg>);
    };


    /// The real part of a diagonal matrix is also diagonal.
    template<typename Scalar, typename Arg>
    struct DiagonalTraits<Eigen::CwiseUnaryView<EGI::scalar_real_ref_op<Scalar>, Arg>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<Arg>;
    };


    /// The imaginary part of a diagonal matrix is also diagonal.
    template<typename Scalar, typename Arg>
    struct DiagonalTraits<Eigen::CwiseUnaryView<EGI::scalar_imag_ref_op<Scalar>, Arg>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<Arg>;
    };


#ifdef __cpp_concepts
    template<typename UnaryOp, triangular_matrix Arg> requires one_by_one_matrix<Arg> or
      detail::triangular_preserving_unary_op<UnaryOp>::value
    struct TriangularTraits<Eigen::CwiseUnaryView<UnaryOp, Arg>>
#else
    template<typename UnaryOp, typename Arg>
    struct TriangularTraits<Eigen::CwiseUnaryView<UnaryOp, Arg>, std::enable_if_t<triangular_matrix<Arg> and
      (one_by_one_matrix<Arg> or detail::triangular_preserving_unary_op<UnaryOp>::value)>>
#endif
    {
      static constexpr TriangleType triangle_type = triangle_type_of_v<Arg>;
      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<typename UnaryOp, hermitian_matrix Arg> requires one_by_one_matrix<Arg> or
      (not diagonal_matrix<Arg> and detail::hermitian_preserving_unary_op<UnaryOp>::value) or
      (diagonal_matrix<Arg> and detail::triangular_preserving_unary_op<UnaryOp>::value)
    struct HermitianTraits<Eigen::CwiseUnaryView<UnaryOp, Arg>>
#else
    template<typename UnaryOp, typename Arg>
    struct HermitianTraits<Eigen::CwiseUnaryView<UnaryOp, Arg>, std::enable_if_t<hermitian_matrix<Arg> and
      (one_by_one_matrix<Arg> or
      (not diagonal_matrix<Arg> and detail::hermitian_preserving_unary_op<UnaryOp>::value) or
      (diagonal_matrix<Arg> and detail::triangular_preserving_unary_op<UnaryOp>::value))>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


    // ---------- //
    //  Diagonal  //
    // ---------- //

    template<typename MatrixType, int DiagIndex>
    struct Dependencies<Eigen::Diagonal<MatrixType, DiagIndex>>
    {
      static constexpr bool has_runtime_parameters = DiagIndex == Eigen::DynamicIndex;

      using type = std::tuple<typename EGI::ref_selector<MatrixType>::non_const_type>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      // Rely on default for convert_to_self_contained. Should always convert to a dense, writable matrix.

    };


    /// The diagonal of a constant matrix is constant.
#ifdef __cpp_concepts
    template<constant_matrix MatrixType, int DiagIndex>
    struct SingleConstant<Eigen::Diagonal<MatrixType, DiagIndex>>
#else
    template<typename MatrixType, int DiagIndex>
    struct SingleConstant<Eigen::Diagonal<MatrixType, DiagIndex>, std::enable_if_t<constant_matrix<MatrixType>>>
#endif
      : SingleConstant<std::decay_t<MatrixType>> {};


    /// The main diagonal of a constant-diagonal matrix is constant; other diagonals are constant 0.
#ifdef __cpp_concepts
    template<constant_diagonal_matrix MatrixType, int DiagIndex> requires (not constant_matrix<MatrixType>) and
      (DiagIndex != Eigen::DynamicIndex)
    struct SingleConstant<Eigen::Diagonal<MatrixType, DiagIndex>>
#else
    template<typename MatrixType, int DiagIndex>
    struct SingleConstant<Eigen::Diagonal<MatrixType, DiagIndex>, std::enable_if_t<
      constant_diagonal_matrix<MatrixType> and (not constant_matrix<MatrixType>) and
      (DiagIndex != Eigen::DynamicIndex)>>
#endif
    {
      static constexpr auto value = DiagIndex == 0 ? constant_diagonal_coefficient_v<MatrixType> : 0;
    };


    /// The main diagonal of an Eigen identity matrix is constant 1; other diagonals are constant 0.
#ifdef __cpp_concepts
    template<eigen_Identity MatrixType, int DiagIndex> requires (not constant_matrix<MatrixType>) and
      (not constant_diagonal_matrix<MatrixType>) and (DiagIndex != Eigen::DynamicIndex)
    struct SingleConstant<Eigen::Diagonal<MatrixType, DiagIndex>>
#else
    template<typename MatrixType, int DiagIndex>
    struct SingleConstant<Eigen::Diagonal<MatrixType, DiagIndex>, std::enable_if_t<eigen_Identity<MatrixType> and
      (not constant_matrix<MatrixType>) and (not constant_diagonal_matrix<MatrixType>) and
      DiagIndex != Eigen::DynamicIndex>>
#endif
    {
      static constexpr auto value = DiagIndex == 0 ? 1 : 0;
    };


    /// The diagonal of a constant, one-by-one matrix is constant-diagonal.
#ifdef __cpp_concepts
    template<constant_matrix MatrixType, int DiagIndex> requires (DiagIndex != Eigen::DynamicIndex) and
      (not dynamic_rows<MatrixType> or (not dynamic_columns<MatrixType> and column_dimension_of_v<MatrixType> == 1)) and
      (not dynamic_columns<MatrixType> or (not dynamic_rows<MatrixType> and row_dimension_of_v<MatrixType> == 1)) and
      (has_dynamic_dimensions<MatrixType> or std::min(row_dimension_of_v<MatrixType> + std::min(DiagIndex, 0),
          column_dimension_of_v<MatrixType> - std::max(DiagIndex, 0)) == 1)
    struct SingleConstantDiagonal<Eigen::Diagonal<MatrixType, DiagIndex>>
#else
    template<typename MatrixType, int DiagIndex>
    struct SingleConstantDiagonal<Eigen::Diagonal<MatrixType, DiagIndex>, std::enable_if_t<
      constant_matrix<MatrixType> and (DiagIndex != Eigen::DynamicIndex) and
      (not dynamic_rows<MatrixType> or (not dynamic_columns<MatrixType> and column_dimension_of<MatrixType>::value == 1)) and
      (not dynamic_columns<MatrixType> or (not dynamic_rows<MatrixType> and row_dimension_of<MatrixType>::value == 1)) and
      (has_dynamic_dimensions<MatrixType> or std::min(row_dimension_of<MatrixType>::value + std::min(DiagIndex, 0),
          column_dimension_of<MatrixType>::value - std::max(DiagIndex, 0)) == 1)>>
#endif
      : SingleConstant<std::decay_t<MatrixType>> {};


    /// The diagonal of a constant-diagonal matrix is constant-diagonal if the result is one-by-one
#ifdef __cpp_concepts
    template<constant_diagonal_matrix MatrixType, int DiagIndex>
    requires (not constant_matrix<MatrixType>) and (DiagIndex != Eigen::DynamicIndex) and
      (not dynamic_rows<MatrixType> or (not dynamic_columns<MatrixType> and column_dimension_of_v<MatrixType> == 1)) and
      (not dynamic_columns<MatrixType> or (not dynamic_rows<MatrixType> and row_dimension_of_v<MatrixType> == 1)) and
      (has_dynamic_dimensions<MatrixType> or std::min(row_dimension_of_v<MatrixType> + std::min(DiagIndex, 0),
          column_dimension_of_v<MatrixType> - std::max(DiagIndex, 0)) == 1)
    struct SingleConstantDiagonal<Eigen::Diagonal<MatrixType, DiagIndex>>
#else
    template<typename MatrixType, int DiagIndex>
    struct SingleConstantDiagonal<Eigen::Diagonal<MatrixType, DiagIndex>, std::enable_if_t<
      constant_diagonal_matrix<MatrixType> and (not constant_matrix<MatrixType>) and
      (DiagIndex != Eigen::DynamicIndex) and
      (not dynamic_rows<MatrixType> or (not dynamic_columns<MatrixType> and column_dimension_of<MatrixType>::value == 1)) and
      (not dynamic_columns<MatrixType> or (not dynamic_rows<MatrixType> and row_dimension_of<MatrixType>::value == 1)) and
      (has_dynamic_dimensions<MatrixType> or std::min(row_dimension_of<MatrixType>::value + std::min(DiagIndex, 0),
          column_dimension_of<MatrixType>::value - std::max(DiagIndex, 0)) == 1)>>
#endif
    {
      static constexpr auto value = DiagIndex == 0 ? constant_diagonal_coefficient_v<MatrixType> : 0;
    };


  // ---------------- //
  //  DiagonalMatrix  //
  // ---------------- //

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
        auto d {make_self_contained(std::forward<Arg>(arg).diagonal())};
        return DiagonalMatrix<decltype(d)> {d};
      }
    };


    template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
    struct DiagonalTraits<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    {
      static constexpr bool is_diagonal = true;
    };


    template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
    struct TriangularTraits<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    {
      static constexpr TriangleType triangle_type = TriangleType::diagonal;
      static constexpr bool is_triangular_adapter = false;
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
        auto d {make_self_contained(std::forward<Arg>(arg).diagonal())};
        return DiagonalMatrix<decltype(d)> {d};
      }
    };


    // A diagonal wrapper is constant if its nested vector is constant and is either zero or has one row.
#ifdef __cpp_concepts
    template<constant_matrix DiagonalVectorType> requires zero_matrix<DiagonalVectorType> or
      (index_dimension_of_v<DiagonalVectorType, 0> == 1 and index_dimension_of_v<DiagonalVectorType, 1> == 1)
    struct SingleConstant<Eigen::DiagonalWrapper<DiagonalVectorType>>
#else
    template<typename DiagonalVectorType>
    struct SingleConstant<Eigen::DiagonalWrapper<DiagonalVectorType>, std::enable_if_t<
      constant_matrix<DiagonalVectorType> and (zero_matrix<DiagonalVectorType> or
      (index_dimension_of_v<DiagonalVectorType, 0> == 1 and index_dimension_of_v<DiagonalVectorType, 1> == 1))>>
#endif
      : SingleConstant<std::decay_t<DiagonalVectorType>> {};


    // A diagonal wrapper is constant-diagonal if its nested vector is constant.
#ifdef __cpp_concepts
    template<constant_matrix DiagonalVectorType>
    struct SingleConstantDiagonal<Eigen::DiagonalWrapper<DiagonalVectorType>>
#else
    template<typename DiagonalVectorType>
    struct SingleConstantDiagonal<Eigen::DiagonalWrapper<DiagonalVectorType>, std::enable_if_t<
      constant_matrix<DiagonalVectorType>>>
#endif
      : SingleConstant<std::decay_t<DiagonalVectorType>> {};


    template<typename DiagonalVectorType>
    struct DiagonalTraits<Eigen::DiagonalWrapper<DiagonalVectorType>>
    {
      static constexpr bool is_diagonal = true;
    };


    template<typename DiagonalVectorType>
    struct TriangularTraits<Eigen::DiagonalWrapper<DiagonalVectorType>>
    {
      static constexpr TriangleType triangle_type = TriangleType::diagonal;
      static constexpr bool is_triangular_adapter = false;
    };

  } // namespace interface


  /**
   * \internal
   * \brief Matrix traits for Eigen::DiagonalWrapper.
   */
  template<typename V>
  struct MatrixTraits<Eigen::DiagonalWrapper<V>>
    : MatrixTraits<Eigen::Matrix<typename EGI::traits<std::decay_t<V>>::Scalar,
        V::SizeAtCompileTime, V::SizeAtCompileTime>>
  {
  };


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
        using N = Eigen::Inverse<equivalent_self_contained_t<XprType>>;
        if constexpr (not std::is_lvalue_reference_v<typename N::XprTypeNested>)
          return N {make_self_contained(arg.nestedExpression())};
        else
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
      }
    };


    // ----- //
    //  Map  //
    // ----- //

    template<typename PlainObjectType, int MapOptions, typename StrideType>
    struct Dependencies<Eigen::Map<PlainObjectType, MapOptions, StrideType>>
    {
    private:
      using M = Eigen::Map<PlainObjectType, MapOptions, StrideType>;
    public:
      static constexpr bool has_runtime_parameters =
        M::RowsAtCompileTime == Eigen::Dynamic or M::ColsAtCompileTime == Eigen::Dynamic or
        M::OuterStrideAtCompileTime == Eigen::Dynamic or M::InnerStrideAtCompileTime == Eigen::Dynamic;

      // Map is not self-contained in any circumstances.
      using type = std::tuple<decltype(*std::declval<typename M::PointerType>())>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return *std::forward<Arg>(arg).data();
      }
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
        using N = Eigen::MatrixWrapper<equivalent_self_contained_t<XprType>>;
        if constexpr (not std::is_lvalue_reference_v<typename N::NestedExpressionType>)
          return N {make_self_contained(arg.nestedExpression())};
        else
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
      }
    };


    // A matrix wrapper is constant if its nested expression is constant.
#ifdef __cpp_concepts
    template<constant_matrix XprType>
    struct SingleConstant<Eigen::MatrixWrapper<XprType>>
#else
    template<typename XprType>
    struct SingleConstant<Eigen::MatrixWrapper<XprType>, std::enable_if_t<constant_matrix<XprType>>>
#endif
      : SingleConstant<std::decay_t<XprType>> {};


    // A matrix wrapper is constant-diagonal if its nested expression is constant-diagonal.
#ifdef __cpp_concepts
    template<constant_diagonal_matrix XprType>
    struct SingleConstantDiagonal<Eigen::MatrixWrapper<XprType>>
#else
    template<typename XprType>
    struct SingleConstantDiagonal<Eigen::MatrixWrapper<XprType>, std::enable_if_t<
      constant_diagonal_matrix<XprType>>>
#endif
      : SingleConstantDiagonal<std::decay_t<XprType>> {};


    template<typename XprType>
    struct DiagonalTraits<Eigen::MatrixWrapper<XprType>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<XprType>;
    };


#ifdef __cpp_concepts
    template<triangular_matrix XprType>
    struct TriangularTraits<Eigen::MatrixWrapper<XprType>>
#else
    template<typename XprType>
    struct TriangularTraits<Eigen::MatrixWrapper<XprType>, std::enable_if_t<triangular_matrix<XprType>>>
#endif
    {
      static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;
      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<hermitian_matrix XprType>
    struct HermitianTraits<Eigen::MatrixWrapper<XprType>>
#else
    template<typename XprType>
    struct HermitianTraits<Eigen::MatrixWrapper<XprType>, std::enable_if_t<hermitian_matrix<XprType>>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


  // ------------------ //
  //  PartialReduxExpr  //
  // ------------------ //

    template<typename MatrixType, typename MemberOp, int Direction>
    struct Dependencies<Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>>
    {
      static constexpr bool has_runtime_parameters = false;

      using type = std::tuple<typename MatrixType::Nested, const MemberOp>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        if constexpr (i == 0)
          return std::forward<Arg>(arg).nestedExpression();
        else
          return std::forward<Arg>(arg).functor();
        static_assert(i <= 1);
      }

      // If a partial redux expression needs to be partially evaluated, it's probably faster to do a full evaluation.
      // Thus, we omit the conversion function.
    };


#ifdef __cpp_concepts
    template<constant_matrix MatrixType, int Direction, int p, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_lpnorm<p, Args...>, Direction>>
#else
    template<typename MatrixType, int Direction, int p, typename...LpnormArgs>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_lpnorm<p, LpnormArgs...>, Direction>,
      std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    {
      static constexpr auto value = (constant_coefficient_v<MatrixType> >= 0) ?
        constant_coefficient_v<MatrixType> : -constant_coefficient_v<MatrixType>;
    };


    // \todo add SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_redux<Args...>, Direction>>


#if not EIGEN_VERSION_AT_LEAST(3,4,0)
#ifdef __cpp_concepts
    template<constant_matrix MatrixType, int Direction, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_squaredNorm<Args...>, Direction>>
#else
    template<typename MatrixType, int Direction, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_squaredNorm<Args...>, Direction>,
      std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    {
      static constexpr auto value = (constant_coefficient_v<MatrixType> >= 0) ?
        constant_coefficient_v<MatrixType> : -constant_coefficient_v<MatrixType>;
    };
#endif


#ifdef __cpp_concepts
    template<constant_matrix MatrixType, int Direction, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_norm<Args...>, Direction>>
#else
    template<typename MatrixType, int Direction, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_norm<Args...>, Direction>,
      std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    {
      static constexpr auto value = (constant_coefficient_v<MatrixType> >= 0) ?
        constant_coefficient_v<MatrixType> : -constant_coefficient_v<MatrixType>;
    };


#ifdef __cpp_concepts
    template<constant_matrix MatrixType, int Direction, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_stableNorm<Args...>, Direction>>
#else
    template<typename MatrixType, int Direction, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_stableNorm<Args...>, Direction>,
      std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    {
      static constexpr auto value = (constant_coefficient_v<MatrixType> >= 0) ?
        constant_coefficient_v<MatrixType> : -constant_coefficient_v<MatrixType>;
    };


#ifdef __cpp_concepts
template<constant_matrix MatrixType, int Direction, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_hypotNorm<Args...>, Direction>>
#else
    template<typename MatrixType, int Direction, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_hypotNorm<Args...>, Direction>,
      std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    {
      static constexpr auto value = (constant_coefficient_v<MatrixType> >= 0) ?
        constant_coefficient_v<MatrixType> : -constant_coefficient_v<MatrixType>;
    };


#ifdef __cpp_concepts
    template<constant_matrix MatrixType, int Direction, typename...Args>
    requires (Direction == Eigen::Horizontal or not dynamic_rows<MatrixType>) and
      (Direction == Eigen::Vertical or not dynamic_columns<MatrixType>)
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_sum<Args...>, Direction>>
#else
      template<typename MatrixType, int Direction, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_sum<Args...>, Direction>,
      std::enable_if_t<constant_matrix<MatrixType> and
        (Direction == Eigen::Horizontal or not dynamic_rows<MatrixType>) and
        (Direction == Eigen::Vertical or not dynamic_columns<MatrixType>)>>
#endif
    {
      static constexpr auto value = constant_coefficient_v<MatrixType> *
        (Direction == Eigen::Vertical ? row_dimension_of_v<MatrixType> : column_dimension_of_v<MatrixType>);
    };


#if not EIGEN_VERSION_AT_LEAST(3,4,0)
#ifdef __cpp_concepts
    template<constant_matrix MatrixType, int Direction, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_mean<Args...>, Direction>>
#else
    template<typename MatrixType, int Direction, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_mean<Args...>, Direction>,
      std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    {
      static constexpr auto value = constant_coefficient_v<MatrixType>;
    };
#endif


#ifdef __cpp_concepts
    template<constant_matrix MatrixType, int Direction, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_minCoeff<Args...>, Direction>>
#else
    template<typename MatrixType, int Direction, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_minCoeff<Args...>, Direction>,
      std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    {
      static constexpr auto value = constant_coefficient_v<MatrixType>;
    };


#ifdef __cpp_concepts
    template<constant_matrix MatrixType, int Direction, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_maxCoeff<Args...>, Direction>>
#else
    template<typename MatrixType, int Direction, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_maxCoeff<Args...>, Direction>,
      std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    {
      static constexpr auto value = constant_coefficient_v<MatrixType>;
    };


#ifdef __cpp_concepts
    template<constant_matrix MatrixType, int Direction, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_all<Args...>, Direction>>
#else
    template<typename MatrixType, int Direction, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_all<Args...>, Direction>,
      std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    {
      static constexpr auto value = constant_coefficient_v<MatrixType>;
    };


#ifdef __cpp_concepts
    template<constant_matrix MatrixType, int Direction, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_any<Args...>, Direction>>
#else
    template<typename MatrixType, int Direction, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_any<Args...>, Direction>,
      std::enable_if_t<constant_matrix<MatrixType>>>
#endif
    {
      static constexpr auto value = constant_coefficient_v<MatrixType>;
    };


#ifdef __cpp_concepts
    template<constant_matrix MatrixType, int Direction, typename...Args>
    requires (Direction == Eigen::Horizontal or not dynamic_rows<MatrixType>) and
      (Direction == Eigen::Vertical or not dynamic_columns<MatrixType>)
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_count<Args...>, Direction>>
#else
    template<typename MatrixType, int Direction, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_count<Args...>, Direction>,
      std::enable_if_t<constant_matrix<MatrixType> and
        (Direction == Eigen::Horizontal or not dynamic_rows<MatrixType>) and
        (Direction == Eigen::Vertical or not dynamic_columns<MatrixType>)>>
#endif
    {
      static constexpr auto value =
        Direction == Eigen::Vertical ? row_dimension_of_v<MatrixType> : column_dimension_of_v<MatrixType>;
    };


#ifdef __cpp_concepts
template<constant_matrix MatrixType, int Direction, typename...Args>
    requires (Direction == Eigen::Horizontal or not dynamic_rows<MatrixType>) and
      (Direction == Eigen::Vertical or not dynamic_columns<MatrixType>)
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_prod<Args...>, Direction>>
#else
    template<typename MatrixType, int Direction, typename...Args>
    struct SingleConstant<Eigen::PartialReduxExpr<MatrixType, EGI::member_prod<Args...>, Direction>,
      std::enable_if_t<constant_matrix<MatrixType> and
        (Direction == Eigen::Horizontal or not dynamic_rows<MatrixType>) and
        (Direction == Eigen::Vertical or not dynamic_columns<MatrixType>)>>
#endif
    {
      static constexpr auto value = OpenKalman::internal::constexpr_pow(constant_coefficient_v<MatrixType>,
        (Direction == Eigen::Vertical ? row_dimension_of_v<MatrixType> : column_dimension_of_v<MatrixType>));
    };


    // A constant partial redux expression is constant-diagonal if it is one-by-one.
#ifdef __cpp_concepts
    template<constant_matrix MatrixType, typename MemberOp, int Direction>
    requires (Direction == Eigen::Vertical and index_dimension_of_v<MatrixType, 1> == 1) or
      (Direction == Eigen::Horizontal and index_dimension_of_v<MatrixType, 0> == 1)
    struct SingleConstantDiagonal<Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>>
#else
    template<typename MatrixType, typename MemberOp, int Direction>
    struct SingleConstantDiagonal<Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>,
      std::enable_if_t<constant_matrix<MatrixType> and
        ((Direction == Eigen::Vertical and column_dimension_of<MatrixType>::value == 1) or
          (Direction == Eigen::Horizontal and row_dimension_of<MatrixType>::value == 1))>>
#endif
      : SingleConstant<std::decay_t<Eigen::PartialReduxExpr<MatrixType, MemberOp, Direction>>> {};


    // ------------------- //
    //  PermutationMatrix  //
    // ------------------- //

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
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
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
        using N = Eigen::Product<equivalent_self_contained_t<LhsType>, equivalent_self_contained_t<RhsType>, Option>;
        constexpr Eigen::Index to_be_evaluated_size = self_contained<LhsType> ?
          RhsType::RowsAtCompileTime * RhsType::ColsAtCompileTime :
          LhsType::RowsAtCompileTime * LhsType::ColsAtCompileTime;

        // Do a partial evaluation if at least one argument is self-contained and result size > non-self-contained size.
        if constexpr ((self_contained<LhsType> or self_contained<RhsType>) and
          (LhsType::RowsAtCompileTime != Eigen::Dynamic) and
          (LhsType::ColsAtCompileTime != Eigen::Dynamic) and
          (RhsType::RowsAtCompileTime != Eigen::Dynamic) and
          (RhsType::ColsAtCompileTime != Eigen::Dynamic) and
          ((Eigen::Index)LhsType::RowsAtCompileTime * (Eigen::Index)RhsType::ColsAtCompileTime > to_be_evaluated_size) and
          not std::is_lvalue_reference_v<typename N::LhsNested> and
          not std::is_lvalue_reference_v<typename N::RhsNested>)
        {
          return N {make_self_contained(arg.lhs()), make_self_contained(arg.rhs())};
        }
        else
        {
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
        }
      }
    };


    // A product that includes a zero matrix is zero.
#ifdef __cpp_concepts
    template<typename Arg1, typename Arg2> requires zero_matrix<Arg1> or zero_matrix<Arg2>
    struct SingleConstant<Eigen::Product<Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::Product<Arg1, Arg2>, std::enable_if_t<zero_matrix<Arg1> or zero_matrix<Arg2>>>
#endif
    {
      static constexpr auto value = 0;
    };


    // A constant-diagonal matrix times a constant matrix is constant.
#ifdef __cpp_concepts
    template<constant_diagonal_matrix Arg1, constant_matrix Arg2> requires
      (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>)
    struct SingleConstant<Eigen::Product<Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
      constant_diagonal_matrix<Arg1> and constant_matrix<Arg2> and not zero_matrix<Arg1> and not zero_matrix<Arg2>>>
#endif
    {
      static constexpr auto value = constant_diagonal_coefficient_v<Arg1> * constant_coefficient_v<Arg2>;
    };


    // A constant matrix times a constant-diagonal matrix is constant.
#ifdef __cpp_concepts
    template<constant_matrix Arg1, constant_diagonal_matrix Arg2> requires
      (not constant_diagonal_matrix<Arg1> or not constant_matrix<Arg2>) and
      (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>)
    struct SingleConstant<Eigen::Product<Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
      constant_matrix<Arg1> and constant_diagonal_matrix<Arg2> and
      (not constant_diagonal_matrix<Arg1> or not constant_matrix<Arg2>) and
      not zero_matrix<Arg1> and not zero_matrix<Arg2>>>
#endif
    {
      static constexpr auto value = constant_coefficient_v<Arg1> * constant_diagonal_coefficient_v<Arg2>;
    };


    // The product of two constant matrices is constant if either columns of Arg1 or rows of Arg2 is known.
#ifdef __cpp_concepts
    template<constant_matrix Arg1, constant_matrix Arg2> requires
      (not constant_diagonal_matrix<Arg1>) and (not constant_diagonal_matrix<Arg2>) and
      (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and (not dynamic_columns<Arg1> or not dynamic_rows<Arg2>)
    struct SingleConstant<Eigen::Product<Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct SingleConstant<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
      constant_matrix<Arg1> and constant_matrix<Arg2> and not zero_matrix<Arg1> and not zero_matrix<Arg2> and
      not constant_diagonal_matrix<Arg1> and not constant_diagonal_matrix<Arg2> and
      (not dynamic_columns<Arg1> or not dynamic_rows<Arg2>)>>
#endif
    {
      static constexpr auto value = constant_coefficient_v<Arg1> * constant_coefficient_v<Arg2> *
          (dynamic_columns<Arg1> ? row_dimension_of_v<Arg2> : column_dimension_of_v<Arg1>);
    };


    /// A constant-diagonal matrix times another constant-diagonal matrix is constant-diagonal.
#ifdef __cpp_concepts
    template<constant_diagonal_matrix Arg1, constant_diagonal_matrix Arg2>
    struct SingleConstantDiagonal<Eigen::Product<Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct SingleConstantDiagonal<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
      constant_diagonal_matrix<Arg1> and constant_diagonal_matrix<Arg2>>>
#endif
    {
      static constexpr auto value = constant_diagonal_coefficient_v<Arg1> * constant_diagonal_coefficient_v<Arg2>;
    };


    /// The constant product of two matrices is constant-diagonal if it is square and zero or one-by-one
#ifdef __cpp_concepts
    template<typename Arg1, typename Arg2> requires
      (not constant_diagonal_matrix<Arg1> or not constant_diagonal_matrix<Arg2>) and
      constant_matrix<Eigen::Product<Arg1, Arg2>> and square_matrix<Eigen::Product<Arg1, Arg2>> and
      (zero_matrix<Eigen::Product<Arg1, Arg2>> or row_dimension_of_v<Arg1> == 1 or column_dimension_of_v<Arg2> == 1)
    struct SingleConstantDiagonal<Eigen::Product<Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct SingleConstantDiagonal<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
      (not constant_diagonal_matrix<Arg1> or not constant_diagonal_matrix<Arg2>) and
      constant_matrix<Eigen::Product<Arg1, Arg2>> and square_matrix<Eigen::Product<Arg1, Arg2>> and
      (zero_matrix<Eigen::Product<Arg1, Arg2>> or row_dimension_of<Arg1>::value == 1 or column_dimension_of<Arg2>::value == 1)>>
#endif
      : SingleConstant<Eigen::Product<Arg1, Arg2>> {};


    /// The product of two diagonal matrices is also diagonal.
    template<typename Arg1, typename Arg2>
    struct DiagonalTraits<Eigen::Product<Arg1, Arg2>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<Arg1> and diagonal_matrix<Arg2>;
    };


    /// A diagonal matrix times a triangular matrix is triangular.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg1, triangular_matrix Arg2>
    struct TriangularTraits<Eigen::Product<Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct TriangularTraits<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
      diagonal_matrix<Arg1> and triangular_matrix<Arg2>>>
#endif
    {
      static constexpr TriangleType triangle_type = triangle_type_of_v<Arg2>;
      static constexpr bool is_triangular_adapter = false;
    };


    /// A triangular matrix times a diagonal matrix is triangular.
#ifdef __cpp_concepts
    template<triangular_matrix Arg1, diagonal_matrix Arg2> requires (not diagonal_matrix<Arg1>)
    struct TriangularTraits<Eigen::Product<Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct TriangularTraits<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
      triangular_matrix<Arg1> and diagonal_matrix<Arg2> and not diagonal_matrix<Arg1>>>
#endif
    {
      static constexpr TriangleType triangle_type = triangle_type_of_v<Arg1>;
      static constexpr bool is_triangular_adapter = false;
    };


    /// A constant diagonal matrix times a self-adjoint matrix (or vice versa) is self-adjoint.
#ifdef __cpp_concepts
    template<constant_diagonal_matrix Arg1, hermitian_matrix Arg2>
    struct HermitianTraits<Eigen::Product<Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct HermitianTraits<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
      (constant_diagonal_matrix<Arg1> and hermitian_matrix<Arg2>)>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


    /// A self-adjoint matrix times a constant-diagonal matrix is self-adjoint.
#ifdef __cpp_concepts
    template<hermitian_matrix Arg1, constant_diagonal_matrix Arg2> requires (not constant_diagonal_matrix<Arg1>)
    struct HermitianTraits<Eigen::Product<Arg1, Arg2>>
#else
    template<typename Arg1, typename Arg2>
    struct HermitianTraits<Eigen::Product<Arg1, Arg2>, std::enable_if_t<
      (hermitian_matrix<Arg1> and constant_diagonal_matrix<Arg2> and not constant_diagonal_matrix<Arg1>)>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


  // ----- //
  //  Ref  //
  // ----- //

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

    public:

      static constexpr bool has_runtime_parameters = RowFactor == Eigen::Dynamic or ColFactor == Eigen::Dynamic;
      using type = std::tuple<typename EGI::traits<T>::MatrixTypeNested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        using N = Eigen::Replicate<equivalent_self_contained_t<MatrixType>, RowFactor, ColFactor>;
        if constexpr (not std::is_lvalue_reference_v<typename EGI::traits<N>::MatrixTypeNested>)
          return N {make_self_contained(arg.nestedExpression())};
        else
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
      }
    };


    // A replication of a constant matrix is constant.
#ifdef __cpp_concepts
    template<constant_matrix MatrixType, int RowFactor, int ColFactor>
    struct SingleConstant<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
#else
    template<typename MatrixType, int RowFactor, int ColFactor>
    struct SingleConstant<Eigen::Replicate<MatrixType, RowFactor, ColFactor>, std::enable_if_t<
      constant_matrix<MatrixType>>>
#endif
      : SingleConstant<std::decay_t<MatrixType>> {};


    // A replication of a constant matrix is constant-diagonal if the result is square and zero.
#ifdef __cpp_concepts
    template<typename MatrixType, int RowFactor, int ColFactor> requires (not constant_diagonal_matrix<MatrixType>) and
      (RowFactor > 0) and (ColFactor > 0) and (not dynamic_rows<MatrixType>) and (not dynamic_columns<MatrixType>) and
      (row_dimension_of_v<MatrixType> * RowFactor == column_dimension_of_v<MatrixType> * ColFactor) and zero_matrix<MatrixType>
    struct SingleConstantDiagonal<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
#else
    template<typename MatrixType, int RowFactor, int ColFactor>
    struct SingleConstantDiagonal<Eigen::Replicate<MatrixType, RowFactor, ColFactor>, std::enable_if_t<
      (not constant_diagonal_matrix<MatrixType>) and
      (RowFactor > 0) and (ColFactor > 0) and (not dynamic_rows<MatrixType>) and (not dynamic_columns<MatrixType>) and
      (row_dimension_of<MatrixType>::value * RowFactor == column_dimension_of<MatrixType>::value * ColFactor) and
      zero_matrix<MatrixType>>>
#endif
      : SingleConstant<std::decay_t<MatrixType>> {};


    // A replication of a constant-diagonal matrix is constant-diagonal if it is replicated only once, or is zero and becomes square.
#ifdef __cpp_concepts
    template<constant_diagonal_matrix MatrixType, int RowFactor, int ColFactor>
    requires (RowFactor == 1 and ColFactor == 1) or
      (are_within_tolerance(constant_diagonal_coefficient_v<MatrixType>, 0) and RowFactor == ColFactor and RowFactor > 0)
    struct SingleConstantDiagonal<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
#else
    template<typename MatrixType, int RowFactor, int ColFactor>
    struct SingleConstantDiagonal<Eigen::Replicate<MatrixType, RowFactor, ColFactor>, std::enable_if_t<
      constant_diagonal_matrix<MatrixType> and ((RowFactor == 1 and ColFactor == 1) or
      (are_within_tolerance(constant_diagonal_coefficient<MatrixType>::value, 0) and RowFactor == ColFactor and RowFactor > 0))>>
#endif
      : SingleConstantDiagonal<std::decay_t<MatrixType>> {};


    template<typename MatrixType>
    struct DiagonalTraits<Eigen::Replicate<MatrixType, 1, 1>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<MatrixType>;
    };


#ifdef __cpp_concepts
    template<triangular_matrix MatrixType>
    struct TriangularTraits<Eigen::Replicate<MatrixType, 1, 1>>
#else
    template<typename MatrixType>
    struct TriangularTraits<Eigen::Replicate<MatrixType, 1, 1>, std::enable_if_t<triangular_matrix<MatrixType>>>
#endif
    {
      static constexpr TriangleType triangle_type = triangle_type_of_v<MatrixType>;
      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<hermitian_matrix MatrixType>
    struct HermitianTraits<Eigen::Replicate<MatrixType, 1, 1>>
#else
    template<typename MatrixType>
    struct HermitianTraits<Eigen::Replicate<MatrixType, 1, 1>, std::enable_if_t<hermitian_matrix<MatrixType>>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


  // --------- //
  //  Reverse  //
  // --------- //

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
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
      }
    };


    // The reverse of a constant matrix is constant.
#ifdef __cpp_concepts
    template<constant_matrix MatrixType, int Direction>
    struct SingleConstant<Eigen::Reverse<MatrixType, Direction>>
#else
    template<typename MatrixType, int Direction>
    struct SingleConstant<Eigen::Reverse<MatrixType, Direction>, std::enable_if_t<constant_matrix<MatrixType>>>
#endif
      : SingleConstant<std::decay_t<MatrixType>> {};


    // The reverse of a constant matrix is constant-diagonal if it is square and zero.
#ifdef __cpp_concepts
    template<typename MatrixType, int Direction> requires (not constant_diagonal_matrix<MatrixType>) and
      square_matrix<MatrixType> and zero_matrix<MatrixType>
    struct SingleConstantDiagonal<Eigen::Reverse<MatrixType, Direction>>
#else
    template<typename MatrixType, int Direction>
    struct SingleConstant<Eigen::Reverse<MatrixType, Direction>, std::enable_if_t<
      (not constant_diagonal_matrix<MatrixType>) and square_matrix<MatrixType> and zero_matrix<MatrixType>>>
#endif
      : SingleConstant<std::decay_t<MatrixType>> {};


    // The double reverse of a constant-diagonal matrix, or any reverse of a zero or one-by-one matrix, is constant-diagonal.
#ifdef __cpp_concepts
    template<constant_diagonal_matrix MatrixType, int Direction> requires (Direction == Eigen::BothDirections) or
      one_by_one_matrix<MatrixType> or (are_within_tolerance(constant_diagonal_coefficient_v<MatrixType>, 0))
    struct SingleConstantDiagonal<Eigen::Reverse<MatrixType, Direction>>
#else
    template<typename MatrixType, int Direction>
    struct SingleConstantDiagonal<Eigen::Reverse<MatrixType, Direction>, std::enable_if_t<
      constant_diagonal_matrix<MatrixType> and (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType> or
      are_within_tolerance(constant_diagonal_coefficient<MatrixType>::value, 0))>>
#endif
      : SingleConstantDiagonal<std::decay_t<MatrixType>> {};


    template<typename MatrixType, int Direction>
    struct DiagonalTraits<Eigen::Reverse<MatrixType, Direction>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<MatrixType> and
        (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType>);
    };


#ifdef __cpp_concepts
    template<triangular_matrix MatrixType, int Direction> requires
      (Direction == Eigen::BothDirections) or (one_by_one_matrix<MatrixType>)
    struct TriangularTraits<Eigen::Reverse<MatrixType, Direction>>
#else
    template<typename MatrixType, int Direction>
    struct TriangularTraits<Eigen::Reverse<MatrixType, Direction>, std::enable_if_t<triangular_matrix<MatrixType> and
      (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType>)>>
#endif
    {
      static constexpr TriangleType triangle_type = diagonal_matrix<MatrixType> ? TriangleType::diagonal :
        (lower_triangular_matrix<MatrixType> ? TriangleType::upper : TriangleType::lower);
      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<hermitian_matrix MatrixType, int Direction> requires
      (Direction == Eigen::BothDirections) or (one_by_one_matrix<MatrixType>)
    struct HermitianTraits<Eigen::Reverse<MatrixType, Direction>>
#else
    template<typename MatrixType, int Direction>
    struct HermitianTraits<Eigen::Reverse<MatrixType, Direction>, std::enable_if_t<hermitian_matrix<MatrixType> and
      (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType>)>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


    // -------- //
    //  Select  //
    // -------- //

    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct Dependencies<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
    {
    private:

      using T = Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>;

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
        using N = Eigen::Select<equivalent_self_contained_t<ConditionMatrixType>,
          equivalent_self_contained_t<ThenMatrixType>, equivalent_self_contained_t<ElseMatrixType>>;
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
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
        }
      }
    };


    // --- constant_coefficient --- //

#ifdef __cpp_concepts
    template<constant_matrix ConditionMatrixType, constant_matrix ThenMatrixType, typename ElseMatrixType>
    requires (static_cast<bool>(constant_coefficient_v<ConditionMatrixType>))
    struct SingleConstant<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct SingleConstant<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
      std::enable_if_t<constant_matrix<ConditionMatrixType> and constant_matrix<ThenMatrixType> and
        static_cast<bool>(constant_coefficient<ConditionMatrixType>::value)>>
#endif
      : SingleConstant<std::decay_t<ThenMatrixType>> {};


#ifdef __cpp_concepts
    template<constant_matrix ConditionMatrixType, typename ThenMatrixType, constant_matrix ElseMatrixType>
    requires (not static_cast<bool>(constant_coefficient_v<ConditionMatrixType>))
    struct SingleConstant<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct SingleConstant<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
      std::enable_if_t<constant_matrix<ConditionMatrixType> and constant_matrix<ElseMatrixType> and
        not static_cast<bool>(constant_coefficient<ConditionMatrixType>::value)>>
#endif
      : SingleConstant<std::decay_t<ElseMatrixType>> {};


#ifdef __cpp_concepts
    template<typename ConditionMatrixType, constant_matrix ThenMatrixType, constant_matrix ElseMatrixType>
    requires (not constant_matrix<ConditionMatrixType>) and
      (constant_coefficient_v<ThenMatrixType> == constant_coefficient_v<ElseMatrixType>)
    struct SingleConstant<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct SingleConstant<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
      std::enable_if_t<constant_matrix<ThenMatrixType> and constant_matrix<ElseMatrixType> and
        (not constant_matrix<ConditionMatrixType>) and
        (constant_coefficient<ThenMatrixType>::value == constant_coefficient<ElseMatrixType>::value)>>
#endif
      : SingleConstant<std::decay_t<ThenMatrixType>> {};


    // --- constant_diagonal_coefficient --- //

    /// A constant selection is constant-diagonal if it is square and zero.
#ifdef __cpp_concepts
    template<constant_matrix ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    requires (not constant_diagonal_matrix<ThenMatrixType>) and
      (not constant_diagonal_matrix<ElseMatrixType>) and square_matrix<ConditionMatrixType> and
      zero_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
    struct SingleConstantDiagonal<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct SingleConstantDiagonal<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
      std::enable_if_t<constant_matrix<ConditionMatrixType> and (not constant_diagonal_matrix<ThenMatrixType>) and
      (not constant_diagonal_matrix<ElseMatrixType>) and square_matrix<ConditionMatrixType> and
      zero_matrix<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>>>
#endif
      : SingleConstant<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>> {};


#ifdef __cpp_concepts
    template<constant_matrix ConditionMatrixType, constant_diagonal_matrix ThenMatrixType, typename ElseMatrixType>
    requires (static_cast<bool>(constant_coefficient_v<ConditionMatrixType>))
    struct SingleConstantDiagonal<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct SingleConstantDiagonal<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
      std::enable_if_t<constant_matrix<ConditionMatrixType> and constant_diagonal_matrix<ThenMatrixType> and
        static_cast<bool>(constant_coefficient<ConditionMatrixType>::value)>>
#endif
      : SingleConstantDiagonal<std::decay_t<ThenMatrixType>> {};


#ifdef __cpp_concepts
    template<constant_matrix ConditionMatrixType, typename ThenMatrixType, constant_diagonal_matrix ElseMatrixType>
    requires (not static_cast<bool>(constant_coefficient_v<ConditionMatrixType>))
    struct SingleConstantDiagonal<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct SingleConstantDiagonal<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
      std::enable_if_t<(constant_matrix<ConditionMatrixType>) and constant_diagonal_matrix<ElseMatrixType> and
        not static_cast<bool>(constant_coefficient<ConditionMatrixType>::value)>>
#endif
      : SingleConstantDiagonal<std::decay_t<ElseMatrixType>> {};


#ifdef __cpp_concepts
    template<typename ConditionMatrixType, constant_diagonal_matrix ThenMatrixType,
      constant_diagonal_matrix ElseMatrixType> requires (not constant_matrix<ConditionMatrixType>) and
      (constant_diagonal_coefficient_v<ThenMatrixType> == constant_diagonal_coefficient_v<ElseMatrixType>)
    struct SingleConstantDiagonal<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct SingleConstantDiagonal<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
      std::enable_if_t<constant_diagonal_matrix<ThenMatrixType> and constant_diagonal_matrix<ElseMatrixType> and
        (not constant_matrix<ConditionMatrixType>) and
        (constant_diagonal_coefficient<ThenMatrixType>::value == constant_diagonal_coefficient<ElseMatrixType>::value)>>
#endif
      : SingleConstantDiagonal<std::decay_t<ThenMatrixType>> {};


#ifdef __cpp_concepts
    template<constant_matrix ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct DiagonalTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct DiagonalTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
      std::enable_if_t<constant_matrix<ConditionMatrixType>>>
#endif
    {
      static constexpr bool is_diagonal =
        (diagonal_matrix<ThenMatrixType> and constant_coefficient_v<ConditionMatrixType>) or
        (diagonal_matrix<ElseMatrixType> and not constant_coefficient_v<ConditionMatrixType>);
    };


#ifdef __cpp_concepts
    template<constant_matrix ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType> requires
      (triangular_matrix<ThenMatrixType> and static_cast<bool>(constant_coefficient_v<ConditionMatrixType>)) or
      (triangular_matrix<ElseMatrixType> and not static_cast<bool>(constant_coefficient_v<ConditionMatrixType>))
    struct TriangularTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct TriangularTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>, std::enable_if_t<
      constant_matrix<ConditionMatrixType> and
      ((triangular_matrix<ThenMatrixType> and static_cast<bool>(constant_coefficient<ConditionMatrixType>::value)) or
       (triangular_matrix<ElseMatrixType> and not static_cast<bool>(constant_coefficient<ConditionMatrixType>::value)))>>
#endif
    {
      static constexpr TriangleType triangle_type = static_cast<bool>(constant_coefficient_v<ConditionMatrixType>) ?
        triangle_type_of_v<ThenMatrixType> : triangle_type_of_v<ElseMatrixType>;
      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<constant_matrix ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType> requires
      (hermitian_matrix<ThenMatrixType> and static_cast<bool>(constant_coefficient_v<ConditionMatrixType>)) or
      (hermitian_matrix<ElseMatrixType> and not static_cast<bool>(constant_coefficient_v<ConditionMatrixType>))
    struct HermitianTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct HermitianTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>, std::enable_if_t<
      constant_matrix<ConditionMatrixType> and
      ((hermitian_matrix<ThenMatrixType> and static_cast<bool>(constant_coefficient<ConditionMatrixType>::value)) or
       (hermitian_matrix<ElseMatrixType> and not static_cast<bool>(constant_coefficient<ConditionMatrixType>::value)))>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


#ifdef __cpp_concepts
    template<hermitian_matrix ConditionMatrixType, hermitian_matrix ThenMatrixType,
      hermitian_matrix ElseMatrixType> requires (not constant_matrix<ConditionMatrixType>)
    struct HermitianTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>>
#else
    template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
    struct HermitianTraits<Eigen::Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType>,
      std::enable_if_t<hermitian_matrix<ConditionMatrixType> and hermitian_matrix<ThenMatrixType> and
        hermitian_matrix<ElseMatrixType> and (not constant_matrix<ConditionMatrixType>)>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };


  // ----------------- //
  //  SelfAdjointView  //
  // ----------------- //

    template<typename M, unsigned int UpLo>
    struct Dependencies<Eigen::SelfAdjointView<M, UpLo>>
    {
      static constexpr bool has_runtime_parameters = false;

      using type = std::tuple<typename Eigen::SelfAdjointView<M, UpLo>::MatrixTypeNested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        return make_self_contained(SelfAdjointMatrix {std::forward<Arg>(arg)});
      }
    };


#ifdef __cpp_concepts
    template<constant_matrix MatrixType, unsigned int UpLo>
    requires (not complex_number<scalar_type_of_t<MatrixType>>) or (imaginary_part(constant_coefficient_v<MatrixType>) == 0)
    struct SingleConstant<Eigen::SelfAdjointView<MatrixType, UpLo>>
#else
    namespace detail
    {
      template<typename T, typename = void>
      struct constant_value_is_real : std::false_type {};

      template<typename T>
      struct constant_value_is_real<T, std::enable_if_t<constant_matrix<T>>>
        : std::bool_constant<not complex_number<scalar_type_of_t<T>> or
        imaginary_part(constant_coefficient_v<T>) == 0> {};
    }

    template<typename MatrixType, unsigned int UpLo>
    struct SingleConstant<Eigen::SelfAdjointView<MatrixType, UpLo>, std::enable_if_t<
      detail::constant_value_is_real<MatrixType>::value>>
#endif
      : SingleConstant<std::decay_t<MatrixType>> {};


#ifdef __cpp_concepts
    template<zero_matrix MatrixType, unsigned int UpLo> requires (not constant_diagonal_matrix<MatrixType>)
    struct SingleConstantDiagonal<Eigen::SelfAdjointView<MatrixType, UpLo>>
#else
    template<typename MatrixType, unsigned int UpLo>
    struct SingleConstantDiagonal<Eigen::SelfAdjointView<MatrixType, UpLo>, std::enable_if_t<
      zero_matrix<MatrixType> and (not constant_diagonal_matrix<MatrixType>)>>
#endif
      : SingleConstant<std::decay_t<MatrixType>> {};


#ifdef __cpp_concepts
    template<constant_diagonal_matrix MatrixType, unsigned int UpLo>
    struct SingleConstantDiagonal<Eigen::SelfAdjointView<MatrixType, UpLo>>
#else
    template<typename MatrixType, unsigned int UpLo>
    struct SingleConstantDiagonal<Eigen::SelfAdjointView<MatrixType, UpLo>, std::enable_if_t<
      constant_diagonal_matrix<MatrixType>>>
#endif
      : SingleConstantDiagonal<std::decay_t<MatrixType>> {};


    template<typename MatrixType, unsigned int UpLo>
    struct DiagonalTraits<Eigen::SelfAdjointView<MatrixType, UpLo>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<MatrixType>;
    };


    template<typename MatrixType, unsigned int UpLo>
    struct TriangularTraits<Eigen::SelfAdjointView<MatrixType, UpLo>>
    {
      static constexpr TriangleType triangle_type = diagonal_matrix<MatrixType> ? TriangleType::diagonal : TriangleType::none;
      static constexpr bool is_triangular_adapter = false;

      template<TriangleType t, typename Arg>
      static constexpr auto make_triangular_matrix(Arg&& arg)
      {
        constexpr auto TriMode = t == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
        return std::forward<Arg>(arg).nestedExpression().template triangularView<TriMode>();
      }
    };


#ifndef __cpp_concepts
    namespace detail
    {
      template<typename T, typename = void>
      struct any_const_imag_part_is_zero : std::false_type {};

      template<typename T>
      struct any_const_imag_part_is_zero<T, std::enable_if_t<
        imaginary_part(constant_coefficient<std::decay_t<T>>::value) != 0>> : std::false_type {};

      template<typename T>
      struct any_const_imag_part_is_zero<T, std::enable_if_t<not constant_matrix<T> and
        imaginary_part(constant_diagonal_coefficient<std::decay_t<T>>::value) != 0>> : std::false_type {};
    };
#endif


#ifdef __cpp_concepts
    template<typename MatrixType, unsigned int UpLo> requires
      (not complex_number<typename EGI::traits<MatrixType>::Scalar>) or
      (imaginary_part(constant_coefficient_v<MatrixType>) == 0) or
      (imaginary_part(constant_diagonal_coefficient_v<MatrixType>) == 0)
    struct HermitianTraits<Eigen::SelfAdjointView<MatrixType, UpLo>>
#else
    template<typename MatrixType, unsigned int UpLo>
    struct HermitianTraits<Eigen::SelfAdjointView<MatrixType, UpLo>, std::enable_if_t<
      (not complex_number<typename EGI::traits<MatrixType>::Scalar>) or
      detail::any_const_imag_part_is_zero<MatrixType>::value>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = diagonal_matrix<MatrixType> ? TriangleType::diagonal :
        (UpLo & Eigen::Upper) != 0 ? TriangleType::upper : TriangleType::lower;

      // make_hermitian_adapter not included because SelfAdjointView is already hermitian.

    };

  } // namespace interface


  /**
   * \brief Deduction guide for converting Eigen::SelfAdjointView to SelfAdjointMatrix
   */
#ifdef __cpp_concepts
  template<Eigen3::eigen_SelfAdjointView M>
#else
  template<typename M, std::enable_if_t<Eigen3::eigen_SelfAdjointView<M>, int> = 0>
#endif
  SelfAdjointMatrix(M&&) -> SelfAdjointMatrix<nested_matrix_of_t<M>, hermitian_adapter_type_of_v<M>>;


  /**
   * \internal
   * \brief Matrix traits for Eigen::SelfAdjointView.
   */
  template<typename M, unsigned int UpLo>
  struct MatrixTraits<Eigen::SelfAdjointView<M, UpLo>> : MatrixTraits<std::decay_t<M>>
  {
  private:

    using Scalar = typename EGI::traits<Eigen::SelfAdjointView<M, UpLo>>::Scalar;
    static constexpr auto rows = row_dimension_of_v<M>;
    static constexpr auto columns = column_dimension_of_v<M>;

    static constexpr TriangleType storage_triangle = UpLo & Eigen::Upper ? TriangleType::upper : TriangleType::lower;

  public:

    template<TriangleType storage_triangle = storage_triangle, std::size_t dim = rows>
    using SelfAdjointMatrixFrom = typename MatrixTraits<std::decay_t<M>>::template SelfAdjointMatrixFrom<storage_triangle, dim>;


    template<TriangleType triangle_type = storage_triangle, std::size_t dim = rows>
    using TriangularMatrixFrom = typename MatrixTraits<std::decay_t<M>>::template TriangularMatrixFrom<triangle_type, dim>;


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
      using type = std::tuple<typename EGI::ref_selector<MatrixType>::non_const_type>;

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
        if constexpr (not std::is_lvalue_reference_v<typename EGI::ref_selector<M>::non_const_type>)
          return N {make_self_contained(arg.nestedExpression())};
        else
          return make_dense_writable_matrix_from(std::forward<Arg>(arg));
      }
    };


#ifdef __cpp_concepts
    template<constant_matrix MatrixType>
    struct SingleConstant<Eigen::Transpose<MatrixType>>
#else
    template<typename MatrixType>
    struct SingleConstant<Eigen::Transpose<MatrixType>, std::enable_if_t<constant_matrix<MatrixType>>>
#endif
      : SingleConstant<std::decay_t<MatrixType>> {};


#ifdef __cpp_concepts
    template<constant_matrix MatrixType> requires square_matrix<MatrixType> and
      (zero_matrix<MatrixType> or row_dimension_of_v<MatrixType> == 1 or column_dimension_of_v<MatrixType> == 1)
    struct SingleConstantDiagonal<Eigen::Transpose<MatrixType>>
#else
    template<typename MatrixType>
    struct SingleConstantDiagonal<Eigen::Transpose<MatrixType>, std::enable_if_t<
      constant_matrix<MatrixType> and square_matrix<MatrixType> and
      (zero_matrix<MatrixType> or row_dimension_of<MatrixType>::value == 1 or column_dimension_of<MatrixType>::value == 1)>>
#endif
      : SingleConstant<std::decay_t<MatrixType>> {};


#ifdef __cpp_concepts
    template<constant_diagonal_matrix MatrixType>
    struct SingleConstantDiagonal<Eigen::Transpose<MatrixType>>
#else
    template<typename MatrixType>
    struct SingleConstantDiagonal<Eigen::Transpose<MatrixType>, std::enable_if_t<
      constant_diagonal_matrix<MatrixType>>>
#endif
      : SingleConstantDiagonal<std::decay_t<MatrixType>> {};


    template<typename MatrixType>
    struct DiagonalTraits<Eigen::Transpose<MatrixType>>
    {
      static constexpr bool is_diagonal = diagonal_matrix<MatrixType>;
    };


#ifdef __cpp_concepts
    template<triangular_matrix MatrixType>
    struct TriangularTraits<Eigen::Transpose<MatrixType>>
#else
    template<typename MatrixType>
    struct TriangularTraits<Eigen::Transpose<MatrixType>, std::enable_if_t<triangular_matrix<MatrixType>>>
#endif
    {
      static constexpr TriangleType triangle_type = diagonal_matrix<MatrixType> ? TriangleType::diagonal :
        (lower_triangular_matrix<MatrixType> ? TriangleType::upper : TriangleType::lower);
      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<hermitian_matrix MatrixType>
    struct HermitianTraits<Eigen::Transpose<MatrixType>>
#else
    template<typename MatrixType>
    struct HermitianTraits<Eigen::Transpose<MatrixType>, std::enable_if_t<hermitian_matrix<MatrixType>>>
#endif
    {
      static constexpr bool is_hermitian = true;
      static constexpr TriangleType adapter_type = TriangleType::none;
    };

  } // namespace interface

    // ---------------- //
    //  TriangularView  //
    // ---------------- //

  /**
   * \brief Deduction guide for converting Eigen::TriangularView to TriangularMatrix
   */
#ifdef __cpp_concepts
  template<Eigen3::eigen_TriangularView M>
#else
  template<typename M, std::enable_if_t<Eigen3::eigen_TriangularView<M>, int> = 0>
#endif
  TriangularMatrix(M&&) -> TriangularMatrix<nested_matrix_of_t<M>, triangle_type_of_v<M>>;


  namespace interface
  {
    template<typename M, unsigned int Mode>
    struct Dependencies<Eigen::TriangularView<M, Mode>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<typename EGI::traits<Eigen::TriangularView<M, Mode>>::MatrixTypeNested>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        return make_self_contained(TriangularMatrix {std::forward<Arg>(arg)});
      }
    };


#ifdef __cpp_concepts
    template<diagonal_matrix MatrixType, unsigned int Mode> requires ((Mode & Eigen::UnitDiag) != 0)
    struct SingleConstant<Eigen::TriangularView<MatrixType, Mode>>
#else
    template<typename MatrixType, unsigned int Mode>
    struct SingleConstant<Eigen::TriangularView<MatrixType, Mode>, std::enable_if_t<
      diagonal_matrix<MatrixType> and (Mode & Eigen::UnitDiag) != 0>>
#endif
    {
      static constexpr auto value = 1;
    };


#ifdef __cpp_concepts
    template<diagonal_matrix MatrixType, unsigned int Mode> requires ((Mode & Eigen::ZeroDiag) != 0)
    struct SingleConstant<Eigen::TriangularView<MatrixType, Mode>>
#else
    template<typename MatrixType, unsigned int Mode>
    struct SingleConstant<Eigen::TriangularView<MatrixType, Mode>, std::enable_if_t<
      diagonal_matrix<MatrixType> and (Mode & Eigen::ZeroDiag) != 0>>
#endif
    {
      static constexpr auto value = 0;
    };


#ifdef __cpp_concepts
    template<constant_matrix MatrixType, unsigned int Mode> requires
      ((Mode & Eigen::UnitDiag) == 0) and ((Mode & Eigen::ZeroDiag) == 0) and
      (zero_matrix<MatrixType> or one_by_one_matrix<MatrixType>)
    struct SingleConstant<Eigen::TriangularView<MatrixType, Mode>>
#else
    template<typename MatrixType, unsigned int Mode>
    struct SingleConstant<Eigen::TriangularView<MatrixType, Mode>, std::enable_if_t<constant_matrix<MatrixType> and
      ((Mode & Eigen::UnitDiag) == 0) and ((Mode & Eigen::ZeroDiag) == 0) and
      (zero_matrix<MatrixType> or one_by_one_matrix<MatrixType>)>>
#endif
      : SingleConstant<std::decay_t<MatrixType>> {};


#ifdef __cpp_concepts
    template<triangular_matrix MatrixType, unsigned int Mode> requires
      (not constant_diagonal_matrix<MatrixType>) and ((Mode & Eigen::UnitDiag) != 0) and
      (((Mode & Eigen::Upper) != 0 and lower_triangular_matrix<MatrixType>) or
        ((Mode & Eigen::Lower) != 0 and upper_triangular_matrix<MatrixType>))
    struct SingleConstantDiagonal<Eigen::TriangularView<MatrixType, Mode>>
#else
    template<typename MatrixType, unsigned int Mode>
    struct SingleConstantDiagonal<Eigen::TriangularView<MatrixType, Mode>, std::enable_if_t<
      triangular_matrix<MatrixType> and (not constant_diagonal_matrix<MatrixType>) and (Mode & Eigen::UnitDiag) != 0 and
      (((Mode & Eigen::Upper) != 0 and lower_triangular_matrix<MatrixType>) or
        ((Mode & Eigen::Lower) != 0 and upper_triangular_matrix<MatrixType>))>>
#endif
    {
      static constexpr auto value = 1;
    };


#ifdef __cpp_concepts
    template<triangular_matrix MatrixType, unsigned int Mode> requires
      (not constant_diagonal_matrix<MatrixType>) and ((Mode & Eigen::ZeroDiag) != 0) and
      (((Mode & Eigen::Upper) != 0 and lower_triangular_matrix<MatrixType>) or
        ((Mode & Eigen::Lower) != 0 and upper_triangular_matrix<MatrixType>))
    struct SingleConstantDiagonal<Eigen::TriangularView<MatrixType, Mode>>
#else
    template<typename MatrixType, unsigned int Mode>
    struct SingleConstantDiagonal<Eigen::TriangularView<MatrixType, Mode>, std::enable_if_t<
      triangular_matrix<MatrixType> and (not constant_diagonal_matrix<MatrixType>) and (Mode & Eigen::ZeroDiag) != 0 and
      (((Mode & Eigen::Upper) != 0 and lower_triangular_matrix<MatrixType>) or
        ((Mode & Eigen::Lower) != 0 and upper_triangular_matrix<MatrixType>))>>
#endif
    {
      static constexpr auto value = 0;
    };


#ifdef __cpp_concepts
    template<typename MatrixType, unsigned int Mode> requires (not constant_diagonal_matrix<MatrixType>) and
      zero_matrix<MatrixType> and ((Mode & Eigen::UnitDiag) == 0) and ((Mode & Eigen::ZeroDiag) == 0)
    struct SingleConstantDiagonal<Eigen::TriangularView<MatrixType, Mode>>
#else
    template<typename MatrixType, unsigned int Mode>
    struct SingleConstantDiagonal<Eigen::TriangularView<MatrixType, Mode>, std::enable_if_t<
      (not constant_diagonal_matrix<MatrixType>) and zero_matrix<MatrixType> and
      ((Mode & Eigen::UnitDiag) == 0) and ((Mode & Eigen::ZeroDiag) == 0)>>
#endif
    {
      static constexpr auto value = 0;
    };


#ifdef __cpp_concepts
    template<constant_diagonal_matrix MatrixType, unsigned int Mode>
    struct SingleConstantDiagonal<Eigen::TriangularView<MatrixType, Mode>>
#else
    template<typename MatrixType, unsigned int Mode>
    struct SingleConstantDiagonal<Eigen::TriangularView<MatrixType, Mode>, std::enable_if_t<
      constant_diagonal_matrix<MatrixType>>>
#endif
    {
      static constexpr auto value = (Mode & Eigen::UnitDiag) != 0 ? 1 :
        ((Mode & Eigen::ZeroDiag) != 0 ? 0 : constant_diagonal_coefficient_v<MatrixType>);
    };


    template<typename MatrixType, unsigned int Mode>
    struct DiagonalTraits<Eigen::TriangularView<MatrixType, Mode>>
    {
      static constexpr bool is_diagonal = ((Mode & Eigen::Lower) != 0 and upper_triangular_matrix<MatrixType>) or
        ((Mode & Eigen::Upper) != 0 and lower_triangular_matrix<MatrixType>);
    };


    template<typename MatrixType, unsigned int Mode>
    struct TriangularTraits<Eigen::TriangularView<MatrixType, Mode>>
    {
      static constexpr TriangleType triangle_type = (Mode & Eigen::Upper) != 0 ?
        (lower_triangular_matrix<MatrixType> ? TriangleType::diagonal : TriangleType::upper) :
        (upper_triangular_matrix<MatrixType> ? TriangleType::diagonal : TriangleType::lower);

      static constexpr bool is_triangular_adapter = true;

      template<TriangleType t, typename Arg>
      static constexpr decltype(auto) make_triangular_matrix(Arg&& arg)
      {
        auto& n = std::forward<Arg>(arg).nestedExpression();
        return TriangularMatrix<decltype(n), TriangleType::diagonal> {n};
      }
    };


#ifdef __cpp_concepts
    template<typename MatrixType, unsigned int Mode> requires
      (not complex_number<typename EGI::traits<MatrixType>::Scalar>) or
      (imaginary_part(constant_coefficient_v<MatrixType>) == 0) or
      (imaginary_part(constant_diagonal_coefficient_v<MatrixType>) == 0)
    struct HermitianTraits<Eigen::TriangularView<MatrixType, Mode>>
#else
    template<typename MatrixType, unsigned int Mode>
    struct HermitianTraits<Eigen::TriangularView<MatrixType, Mode>, std::enable_if_t<
      (not complex_number<typename EGI::traits<MatrixType>::Scalar>) or
      detail::any_const_imag_part_is_zero<MatrixType>::value>>
#endif
    {
      static constexpr bool is_hermitian = diagonal_matrix<MatrixType>;
      static constexpr TriangleType adapter_type = is_hermitian ? TriangleType::diagonal : TriangleType::none;

      template<TriangleType t, typename Arg>
      static constexpr auto make_hermitian_adapter(Arg&& arg)
      {
        static_assert(not hermitian_matrix<Arg>);
        constexpr auto TriMode = t == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
        return std::forward<Arg>(arg).nestedExpression().template selfadjointView<TriMode>();
      }
    };

  } // namespace interface


  /**
   * \internal
   * \brief Matrix traits for Eigen::TriangularView.
   */
  template<typename M, unsigned int Mode>
  struct MatrixTraits<Eigen::TriangularView<M, Mode>> : MatrixTraits<std::decay_t<M>>
  {
  private:

    using Scalar = typename EGI::traits<Eigen::TriangularView<M, Mode>>::Scalar;
    static constexpr auto rows = row_dimension_of_v<M>;
    static constexpr auto columns = column_dimension_of_v<M>;

    static constexpr TriangleType triangle_type = Mode & Eigen::Upper ? TriangleType::upper : TriangleType::lower;

  public:

    template<TriangleType storage_triangle = triangle_type, std::size_t dim = rows>
    using SelfAdjointMatrixFrom = typename MatrixTraits<std::decay_t<M>>::template SelfAdjointMatrixFrom<storage_triangle, dim>;


    template<TriangleType triangle_type = triangle_type, std::size_t dim = rows>
    using TriangularMatrixFrom = typename MatrixTraits<std::decay_t<M>>::template TriangularMatrixFrom<triangle_type, dim>;


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


  // ------------- //
  //  VectorBlock  //
  // ------------- //

  namespace interface
  {

    template<typename VectorType, int Size>
    struct Dependencies<Eigen::VectorBlock<VectorType, Size>>
    {
      static constexpr bool has_runtime_parameters = true;
      using type = std::tuple<typename EGI::ref_selector<VectorType>::non_const_type>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nestedExpression();
      }

      // Eigen::VectorBlock should always be converted to Matrix

    };


    // A segment taken from a constant vector is constant.
#ifdef __cpp_concepts
    template<constant_matrix VectorType, int Size>
    struct SingleConstant<Eigen::VectorBlock<VectorType, Size>>
#else
      template<typename VectorType, int Size>
      struct SingleConstant<Eigen::VectorBlock<VectorType, Size>, std::enable_if_t<constant_matrix<VectorType>>>
#endif
      : SingleConstant<std::decay_t<VectorType>> {};


    // A segment taken from a constant vector is constant-diagonal if it is one-by-one.
#ifdef __cpp_concepts
    template<constant_matrix VectorType, int Size> requires (Size == 1 or one_by_one_matrix<VectorType>)
    struct SingleConstantDiagonal<Eigen::VectorBlock<VectorType, Size>>
#else
    template<typename VectorType, int Size>
    struct SingleConstantDiagonal<Eigen::VectorBlock<VectorType, Size>, std::enable_if_t<
      constant_matrix<VectorType> and (Size == 1 or one_by_one_matrix<VectorType>)>>
#endif
      : SingleConstant<std::decay_t<VectorType>> {};


  // -------------- //
  //  VectorWiseOp  //
  // -------------- //

    template<typename ExpressionType, int Direction>
    struct IndexibleObjectTraits<Eigen::VectorwiseOp<ExpressionType, Direction>>
    {
      static constexpr std::size_t max_indices = 2;
      using scalar_type = typename std::decay_t<Eigen::VectorwiseOp<ExpressionType, Direction>>::Scalar;
    };


    template<typename ExpressionType, int Direction>
    struct IndexTraits<Eigen::VectorwiseOp<ExpressionType, Direction>, Direction == Eigen::Vertical ? 1 : 0>
    {
    private:

      static constexpr std::size_t ix = Direction == Eigen::Vertical ? 1 : 0;

    public:

      static constexpr std::size_t dimension = index_dimension_of_v<ExpressionType, ix>;

      template<typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        return get_index_dimension_of<ix>(arg);
      }
    };


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
        using N = Eigen::VectorwiseOp<equivalent_self_contained_t<ExpressionType>, Direction>;
        static_assert(self_contained<typename N::ExpressionTypeNested>,
          "This VectorWiseOp expression cannot be made self-contained");
        return N {make_self_contained(arg._expression())};
      }
    };


    // A vectorwise operation on a constant vector is constant.
#ifdef __cpp_concepts
    template<constant_matrix ExpressionType, int Direction>
    struct SingleConstant<Eigen::VectorwiseOp<ExpressionType, Direction>>
#else
    template<typename ExpressionType, int Direction>
    struct SingleConstant<Eigen::VectorwiseOp<ExpressionType, Direction>,
      std::enable_if_t<constant_matrix<ExpressionType>>>
#endif
      : SingleConstant<std::decay_t<ExpressionType>> {};

  } // namespace interface

} // namespace OpenKalman

#endif //OPENKALMAN_EIGEN3_TRAITS_HPP
