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
 * \brief Forward declarations for OpenKalman's Eigen interface.
 */

#ifndef OPENKALMAN_EIGEN_FORWARD_DECLARATIONS_HPP
#define OPENKALMAN_EIGEN_FORWARD_DECLARATIONS_HPP

#include <type_traits>


namespace OpenKalman::Eigen3
{

  /**
   * \internal
   * \brief Traits for n-ary functors (n>0).
   * \tparam Operation The n-ary operation.
   * \tparam XprTypes Any argument types.
   * \sa NullaryFunctorTraits
   */
  template<typename Operation, typename...XprTypes>
  struct FunctorTraits
  {
    /**
     * \brief Return a scalar constant or std::monostate
     * \tparam is_diag True if \ref constant_diagonal_coefficient, false if \ref constant_coefficient.
     * \return \ref scalar_constant
     */
    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return std::monostate {};
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = false;
  };


  /**
   * \internal
   * \brief Traits for nullary functors (n>0).
   * \tparam Operation The n-ary operation.
   * \tparam XprTypes Any argument types.
   * \sa FunctorTraits
   */
  template<typename NullaryOp, typename PlainObjectType>
  struct NullaryFunctorTraits;


  /**
   * \internal
   * \brief The ultimate Eigen base for OpenKalman classes.
   * \details This class is used mainly to distinguish OpenKalman classes from native Eigen classes which are
   * also derived from Eigen::MatrixBase or Eigen::ArrayBase.
   */
  struct EigenDenseBase {};


  /**
   * \internal
   * \brief The ultimate base for Eigen-based adapter classes in OpenKalman.
   * \details This class adds base features required by Eigen.
   */
  template<typename Derived, typename NestedMatrix>
  struct EigenAdapterBase;


  // -------------------------------------- //
  //   concepts for specific Eigen classes  //
  // -------------------------------------- //

  namespace detail
  {
    template<typename T>
    struct is_eigen_Block : std::false_type {};

    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    struct is_eigen_Block<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>> : std::true_type {};


    template<typename T>
    struct is_eigen_VectorBlock : std::false_type {};

    template<typename T, int Size>
    struct is_eigen_VectorBlock<Eigen::VectorBlock<T, Size>> : std::true_type {};


    template<typename T>
    struct is_eigen_SelfAdjointView : std::false_type {};

    template<typename MatrixType, unsigned int UpLo>
    struct is_eigen_SelfAdjointView<Eigen::SelfAdjointView<MatrixType, UpLo>> : std::true_type {};


    template<typename T>
    struct is_eigen_TriangularView : std::false_type {};

    template<typename MatrixType, unsigned int UpLo>
    struct is_eigen_TriangularView<Eigen::TriangularView<MatrixType, UpLo>> : std::true_type {};


    template<typename T>
    struct is_eigen_DiagonalMatrix : std::false_type {};

    template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
    struct is_eigen_DiagonalMatrix<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
      : std::true_type {};


    template<typename T>
    struct is_eigen_DiagonalWrapper : std::false_type {};

    template<typename DiagonalVectorType>
    struct is_eigen_DiagonalWrapper<Eigen::DiagonalWrapper<DiagonalVectorType>> : std::true_type {};


    template<typename T>
    struct is_eigen_Identity : std::false_type {};

    template<typename Scalar, typename Arg>
    struct is_eigen_Identity<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<Scalar>, Arg>>
      : std::true_type {};


    template<typename T>
    struct is_eigen_MatrixWrapper : std::false_type {};

    template<typename XprType>
    struct is_eigen_MatrixWrapper<Eigen::MatrixWrapper<XprType>> : std::true_type {};


    template<typename T>
    struct is_eigen_ArrayWrapper : std::false_type {};

    template<typename XprType>
    struct is_eigen_ArrayWrapper<Eigen::ArrayWrapper<XprType>> : std::true_type {};
  }


    /**
     * \brief Specifies whether T is Eigen::Block
     */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_block =
#else
  constexpr bool eigen_block =
#endif
    detail::is_eigen_Block<std::decay_t<T>>::value;


  /**
   * \brief Specifies whether T is Eigen::VectorBlock
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_VectorBlock =
#else
  constexpr bool eigen_VectorBlock =
#endif
    detail::is_eigen_VectorBlock<std::decay_t<T>>::value;


  /**
   * \brief T is of type Eigen::SelfAdjointView.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_SelfAdjointView = detail::is_eigen_SelfAdjointView<std::decay_t<T>>::value;
#else
  constexpr bool eigen_SelfAdjointView = detail::is_eigen_SelfAdjointView<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is of type Eigen::TriangularView.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_TriangularView = detail::is_eigen_TriangularView<std::decay_t<T>>::value;
#else
  constexpr bool eigen_TriangularView = detail::is_eigen_TriangularView<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is of type Eigen::DiagonalMatrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_DiagonalMatrix = detail::is_eigen_DiagonalMatrix<std::decay_t<T>>::value;
#else
  constexpr bool eigen_DiagonalMatrix = detail::is_eigen_DiagonalMatrix<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is of type Eigen::DiagonalMatrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_DiagonalWrapper = detail::is_eigen_DiagonalWrapper<std::decay_t<T>>::value;
#else
  constexpr bool eigen_DiagonalWrapper = detail::is_eigen_DiagonalWrapper<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is an Eigen identity matrix (not necessarily an \ref identity_matrix).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_Identity = detail::is_eigen_Identity<std::decay_t<T>>::value;
#else
  constexpr bool eigen_Identity = detail::is_eigen_Identity<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is of type Eigen::DiagonalMatrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_MatrixWrapper = detail::is_eigen_MatrixWrapper<std::decay_t<T>>::value;
#else
  constexpr bool eigen_MatrixWrapper = detail::is_eigen_MatrixWrapper<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is of type Eigen::DiagonalMatrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_ArrayWrapper = detail::is_eigen_ArrayWrapper<std::decay_t<T>>::value;
#else
  constexpr bool eigen_ArrayWrapper = detail::is_eigen_ArrayWrapper<std::decay_t<T>>::value;
#endif


  /**
   * \brief Specifies any descendant of Eigen::EigenBase.
   * \tparam must_be_native T is required to be a native Eigen object.
   */
  template<typename T, bool must_be_native = false>
#ifdef __cpp_concepts
  concept eigen_general =
#else
  constexpr bool eigen_general =
#endif
    (std::is_base_of_v<Eigen::EigenBase<std::decay_t<T>>, std::decay_t<T>> or eigen_VectorBlock<T>) and
    (not must_be_native or not std::is_base_of_v<EigenDenseBase, std::decay_t<T>>);


  namespace detail
  {
    template<typename T>
    struct is_eigen_matrix_VectorBlock : std::false_type {};

    template<typename T, int Size>
    struct is_eigen_matrix_VectorBlock<Eigen::VectorBlock<T, Size>>
      : std::is_base_of<Eigen::MatrixBase<std::decay_t<T>>, std::decay_t<T>> {};
  }


  /**
   * \brief Specifies a native Eigen3 matrix or expression class deriving from Eigen::MatrixBase.
   * \tparam must_be_native T is required to be a native Eigen object.
   */
  template<typename T, bool must_be_native = false>
#ifdef __cpp_concepts
  concept eigen_matrix_general =
#else
  constexpr bool eigen_matrix_general =
#endif
    (eigen_general<T, must_be_native> and std::is_base_of_v<Eigen::MatrixBase<std::decay_t<T>>, std::decay_t<T>>) or
      detail::is_eigen_matrix_VectorBlock<std::decay_t<T>>::value;


  namespace detail
  {
    template<typename T>
    struct is_eigen_array_VectorBlock : std::false_type {};

    template<typename T, int Size>
    struct is_eigen_array_VectorBlock<Eigen::VectorBlock<T, Size>>
      : std::is_base_of<Eigen::ArrayBase<std::decay_t<T>>, std::decay_t<T>> {};
  }


  /**
   * \brief Specifies a native Eigen3 array or expression class deriving from Eigen::ArrayBase.
   * \tparam must_be_native T is required to be a native Eigen object.
   */
  template<typename T, bool must_be_native = false>
#ifdef __cpp_concepts
  concept eigen_array_general =
#else
  constexpr bool eigen_array_general =
#endif
    (eigen_general<T, must_be_native> and std::is_base_of_v<Eigen::ArrayBase<std::decay_t<T>>, std::decay_t<T>>) or
      detail::is_eigen_array_VectorBlock<std::decay_t<T>>::value;


  /**
   * \brief Specifies a native Eigen3 object deriving from Eigen::MatrixBase or Eigen::ArrayBase.
   * \tparam must_be_native T is required to be a native Eigen object.
   */
  template<typename T, bool must_be_native = false>
#ifdef __cpp_concepts
  concept eigen_dense_general =
#else
  constexpr bool eigen_dense_general =
#endif
    eigen_matrix_general<T, must_be_native> or eigen_array_general<T, must_be_native>;


  /**
   * \brief An alias for the Eigen identity matrix.
   * \note In Eigen, this does not need to be a \ref square_matrix.
   * \tparam NestedMatrix The nested matrix on which the identity is based.
   */
  template<typename NestedMatrix>
  using IdentityMatrix = Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<
    typename Eigen::internal::traits<std::decay_t<NestedMatrix>>::Scalar>, NestedMatrix>;


  namespace detail
  {
    template<typename T>
    struct is_eigen_LibraryWrapper : std::false_type {};

    template<typename NestedObject, typename LibraryObject>
    struct is_eigen_LibraryWrapper<internal::LibraryWrapper<NestedObject, LibraryObject>>
      : std::bool_constant<eigen_general<LibraryObject, true>> {};
  } // namespace detail


  /**
   * \internal
   * \brief T is Eigen3::EigenWrapper<T> or internal::LibraryWrapper for a , for any T.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_wrapper =
#else
  constexpr bool eigen_wrapper =
#endif
    detail::is_eigen_LibraryWrapper<std::decay_t<T>>::value;


  /**
   * \internal
   * \brief Alias for the Eigen version of LibraryWrapper.
   * \details A dumb wrapper for OpenKalman classes so that they are treated exactly as native Eigen types.
   * \tparam NestedObject A non-Eigen class, for which an Eigen3 trait and evaluator is defined.
   */
#ifdef __cpp_concepts
  template<indexible NestedObject> requires (index_count_v<NestedObject> <= 2)
#else
  template<typename NestedObject>
#endif
  using EigenWrapper = internal::LibraryWrapper<NestedObject, Eigen::Matrix<scalar_type_of_t<NestedObject>, 0, 0>>;


  // ---------------- //
  //  eigen_matrix_t  //
  // ---------------- //

  namespace detail
  {
    template<std::size_t size>
    constexpr auto eigen_index_convert = size == dynamic_size ? Eigen::Dynamic : static_cast<Eigen::Index>(size);
  }

  /**
   * \brief An alias for a self-contained, writable, native Eigen matrix.
   * \tparam Scalar Scalar type of the matrix (defaults to the Scalar type of T).
   * \tparam rows Number of rows in the native matrix (0 if not fixed at compile time).
   * \tparam cols Number of columns in the native matrix (0 if not fixed at compile time).
   */
  template<typename Scalar, std::size_t...dims>
  using eigen_matrix_t = std::conditional_t<sizeof...(dims) == 1,
    Eigen::Matrix<Scalar, detail::eigen_index_convert<dims>..., detail::eigen_index_convert<1>>,
    Eigen::Matrix<Scalar, detail::eigen_index_convert<dims>...>>;



  /**
   * \brief Trait object providing get and set routines
   */
#ifdef __cpp_concepts
  template<typename T>
  struct indexible_object_traits_base;
#else
  template<typename T, typename = void>
  struct indexible_object_traits_base;
#endif


  /**
   * \brief Convert std functions (e.g., std::plus) to equivalent Eigen operations for possible vectorization:
   */
  template<typename Op> static decltype(auto) native_operation(Op&& op) { return std::forward<Op>(op); };
  template<typename S> static auto native_operation(const std::plus<S>& op) { return Eigen::internal::scalar_sum_op<S, S> {}; };
  template<typename S> static auto native_operation(const std::minus<S>& op) { return Eigen::internal::scalar_difference_op<S, S> {}; };
  template<typename S> static auto native_operation(const std::multiplies<S>& op) {return Eigen::internal::scalar_product_op<S, S> {}; };
  template<typename S> static auto native_operation(const std::divides<S>& op) { return Eigen::internal::scalar_quotient_op<S, S> {}; };
  template<typename S> static auto native_operation(const std::negate<S>& op) { return Eigen::internal::scalar_opposite_op<S> {}; };

  using EIC = Eigen::internal::ComparisonName;
  template<typename S> static auto native_operation(const std::equal_to<S>& op) { return Eigen::internal::scalar_cmp_op<S, S, EIC::cmp_EQ> {}; };
  template<typename S> static auto native_operation(const std::not_equal_to<S>& op) { return Eigen::internal::scalar_cmp_op<S, S, EIC::cmp_NEQ> {}; };
  template<typename S> static auto native_operation(const std::greater<S>& op) { return Eigen::internal::scalar_cmp_op<S, S, EIC::cmp_GT> {}; };
  template<typename S> static auto native_operation(const std::less<S>& op) { return Eigen::internal::scalar_cmp_op<S, S, EIC::cmp_LT> {}; };
  template<typename S> static auto native_operation(const std::greater_equal<S>& op) { return Eigen::internal::scalar_cmp_op<S, S, EIC::cmp_GE> {}; };
  template<typename S> static auto native_operation(const std::less_equal<S>& op) { return Eigen::internal::scalar_cmp_op<S, S, EIC::cmp_LE> {}; };

  template<typename S> static auto native_operation(const std::logical_and<S>& op) { return Eigen::internal::scalar_boolean_and_op {}; };
  template<typename S> static auto native_operation(const std::logical_or<S>& op) { return Eigen::internal::scalar_boolean_or_op {}; };
  template<typename S> static auto native_operation(const std::logical_not<S>& op) { return Eigen::internal::scalar_boolean_not_op<S> {}; };


} // namespace OpenKalman::Eigen3


#endif //OPENKALMAN_EIGEN_FORWARD_DECLARATIONS_HPP
