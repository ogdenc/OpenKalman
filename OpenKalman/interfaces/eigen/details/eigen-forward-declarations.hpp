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
   * \brief Traits for nullary functors.
   * \tparam Operation The nullary operation.
   * \tparam PlainObjectType A matrix for which the shape of the result is based.
   */
  template<typename NullaryOp, typename PlainObjectType>
  struct NullaryFunctorTraits;


  /**
   * \internal
   * \brief Traits for unary functors.
   * \tparam Operation The unary operation.
   */
  template<typename Operation>
  struct UnaryFunctorTraits;


  /**
   * \internal
   * \brief The type of binary functor.
   */
  enum struct BinaryFunctorType: int {
    normal, ///< A normal binary functor.
    sum, ///< The result of the operation is triangular if the arguments are either both upper- or both lower-triangular.
    product, ///< The result of the operation is triangular if either argument is triangular.
  };


  /**
   * \internal
   * \brief Traits for binary functors.
   * \tparam Operation The binary operation.
   */
  template<typename Operation>
  struct BinaryFunctorTraits;


  /**
   * \internal
   * \brief Traits for ternary functors.
   * \tparam Operation The ternary operation.
   */
  template<typename Operation, typename Arg1, typename Arg2, typename Arg3>
  struct TernaryFunctorTraits;


  /**
   * \internal
   * \brief The ultimate Eigen base for custom Eigen classes.
   * \details This class is used mainly to distinguish OpenKalman classes from native Eigen classes which are
   * also derived from Eigen::MatrixBase or Eigen::ArrayBase.
   */
  struct EigenCustomBase {};


  /**
   * \internal
   * \brief The ultimate base for Eigen-based adapter classes in OpenKalman.
   * \details This class adds base features required by Eigen.
   * \tparam Derived The Derived object
   * \tparam Base The Eigen Base (e.g., <code>Eigen::MatrixBase</code>, <code>Eigen::ArrayBase</code>, etc.)
   */
  template<typename Derived, typename Base>
  struct EigenAdapterBase;


  // -------------------------------------- //
  //   concepts for specific Eigen classes  //
  // -------------------------------------- //

  namespace detail
  {
    template<typename T>
    struct is_ArrayWrapper : std::false_type {};

    template<typename XprType>
    struct is_ArrayWrapper<Eigen::ArrayWrapper<XprType>> : std::true_type {};
  }


  /**
   * \brief T is of type Eigen::ArrayWrapper.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_ArrayWrapper =
#else
  constexpr bool eigen_ArrayWrapper =
#endif
    detail::is_ArrayWrapper<std::decay_t<T>>::value;


  namespace detail
  {
    template<typename T>
    struct is_Block : std::false_type {};

    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    struct is_Block<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>> : std::true_type {};
  }


  /**
   * \brief Specifies whether T is Eigen::Block
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_Block =
#else
  constexpr bool eigen_Block =
#endif
    detail::is_Block<std::decay_t<T>>::value;


  namespace detail
  {
    template<typename T>
    struct is_CwiseUnaryOp : std::false_type {};

    template<typename UnaryOp, typename XprType>
    struct is_CwiseUnaryOp<Eigen::CwiseUnaryOp<UnaryOp, XprType>> : std::true_type {};
  }


  /**
   * \brief Specifies whether T is Eigen::CwiseUnaryOp
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_CwiseUnaryOp =
#else
  constexpr bool eigen_CwiseUnaryOp =
#endif
    detail::is_CwiseUnaryOp<std::decay_t<T>>::value;


  namespace detail
  {
    template<typename T>
    struct is_CwiseUnaryView : std::false_type {};

    template<typename ViewOp, typename XprType>
    struct is_CwiseUnaryView<Eigen::CwiseUnaryView<ViewOp, XprType>> : std::true_type {};
  }


  /**
   * \brief Specifies whether T is Eigen::CwiseUnaryView
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_CwiseUnaryView =
#else
  constexpr bool eigen_CwiseUnaryView =
#endif
  detail::is_CwiseUnaryView<std::decay_t<T>>::value;


  namespace detail
  {
    template<typename T>
    struct is_DiagonalMatrix : std::false_type {};

    template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
    struct is_DiagonalMatrix<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
      : std::true_type {};
  }


  /**
   * \brief T is of type Eigen::DiagonalMatrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_DiagonalMatrix =
#else
  constexpr bool eigen_DiagonalMatrix =
#endif
    detail::is_DiagonalMatrix<std::decay_t<T>>::value;


  namespace detail
  {
    template<typename T>
    struct is_DiagonalWrapper : std::false_type {};

    template<typename DiagonalVectorType>
    struct is_DiagonalWrapper<Eigen::DiagonalWrapper<DiagonalVectorType>> : std::true_type {};
  }


  /**
   * \brief T is of type Eigen::DiagonalMatrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_DiagonalWrapper =
#else
  constexpr bool eigen_DiagonalWrapper =
#endif
    detail::is_DiagonalWrapper<std::decay_t<T>>::value;


  namespace detail
  {
    template<typename T>
    struct is_eigen_Identity : std::false_type {};

    template<typename Scalar, typename Arg>
    struct is_eigen_Identity<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<Scalar>, Arg>>
      : std::true_type {};
  }


  /**
   * \brief T is an Eigen identity matrix (not necessarily an \ref identity_matrix).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_Identity = detail::is_eigen_Identity<std::decay_t<T>>::value;
#else
  constexpr bool eigen_Identity = detail::is_eigen_Identity<std::decay_t<T>>::value;
#endif


  namespace detail
  {
    template<typename T>
    struct is_eigen_MatrixWrapper : std::false_type {};

    template<typename XprType>
    struct is_eigen_MatrixWrapper<Eigen::MatrixWrapper<XprType>> : std::true_type {};
  }


  /**
   * \brief T is of type Eigen::MatrixWrapper.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_MatrixWrapper = detail::is_eigen_MatrixWrapper<std::decay_t<T>>::value;
#else
  constexpr bool eigen_MatrixWrapper = detail::is_eigen_MatrixWrapper<std::decay_t<T>>::value;
#endif


  namespace detail
  {
    template<typename T>
    struct is_eigen_Replicate : std::false_type {};

    template<typename MatrixType, int RowFactor, int ColFactor>
    struct is_eigen_Replicate<Eigen::Replicate<MatrixType, RowFactor, ColFactor>> : std::true_type
    {
    private:

      template<std::size_t direction>
      static constexpr std::size_t efactor = direction == 1 ? ColFactor : RowFactor;

    public:

      template<std::size_t direction>
      static constexpr std::size_t factor = efactor<direction> == Eigen::Dynamic ? dynamic_size : efactor<direction>;
    };
  }


  /**
   * \brief T is of type Eigen::Replicate.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_Replicate =
#else
  constexpr bool eigen_Replicate =
#endif
    detail::is_eigen_Replicate<std::decay_t<T>>::value;


  /**
   * \brief The replication factor for Eigen::Replicate in a given direction.
   */
  template<typename T, std::size_t direction>
  struct eigen_Replicate_factor
    : std::integral_constant<std::size_t, detail::is_eigen_Replicate<std::decay_t<T>>::template factor<direction>> {};


  /**
   * \brief Helper template for eigen_Replicate_factor.
   */
  template<typename T, std::size_t direction>
  constexpr auto eigen_Replicate_factor_v = eigen_Replicate_factor<T, direction>::value;


  namespace detail
  {
    template<typename T>
    struct is_eigen_SelfAdjointView : std::false_type {};

    template<typename MatrixType, unsigned int UpLo>
    struct is_eigen_SelfAdjointView<Eigen::SelfAdjointView<MatrixType, UpLo>> : std::true_type {};
  }


  /**
   * \brief T is of type Eigen::SelfAdjointView.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_SelfAdjointView = detail::is_eigen_SelfAdjointView<std::decay_t<T>>::value;
#else
  constexpr bool eigen_SelfAdjointView = detail::is_eigen_SelfAdjointView<std::decay_t<T>>::value;
#endif


  namespace detail
  {
    template<typename T>
    struct is_eigen_TriangularView : std::false_type {};

    template<typename MatrixType, unsigned int UpLo>
    struct is_eigen_TriangularView<Eigen::TriangularView<MatrixType, UpLo>> : std::true_type {};
  }


  /**
   * \brief T is of type Eigen::TriangularView.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_TriangularView = detail::is_eigen_TriangularView<std::decay_t<T>>::value;
#else
  constexpr bool eigen_TriangularView = detail::is_eigen_TriangularView<std::decay_t<T>>::value;
#endif


  namespace detail
  {
    template<typename T>
    struct is_eigen_VectorBlock : std::false_type {};

    template<typename T, int Size>
    struct is_eigen_VectorBlock<Eigen::VectorBlock<T, Size>> : std::true_type {};
  }


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


  namespace detail
  {
    template<typename T>
    struct is_eigen_SelfContainedWrapper : std::false_type {};

    template<typename BaseObject, typename...InternalizedParameters>
    struct is_eigen_SelfContainedWrapper<OpenKalman::internal::SelfContainedWrapper<BaseObject, InternalizedParameters...>> : std::true_type {};
  }


  /**
   * \brief Specifies whether T is internal::SelfContainedWrapper
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_SelfContainedWrapper =
#else
  constexpr bool eigen_SelfContainedWrapper =
#endif
    detail::is_eigen_SelfContainedWrapper<std::decay_t<T>>::value;


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
    ((std::is_base_of_v<Eigen::EigenBase<std::decay_t<T>>, std::decay_t<T>> or eigen_VectorBlock<T>) and
      (not must_be_native or not std::is_base_of_v<EigenCustomBase, std::decay_t<T>>)) or
    (not must_be_native and eigen_SelfContainedWrapper<T>);


  namespace detail
  {
    template<typename T>
    struct is_derived_eigen_matrix : std::false_type {};

    template<typename T, int Size>
    struct is_derived_eigen_matrix<Eigen::VectorBlock<T, Size>>
      : std::is_base_of<Eigen::MatrixBase<std::decay_t<T>>, std::decay_t<T>> {};

    template<typename BaseObject, typename...InternalizedParameters>
    struct is_derived_eigen_matrix<OpenKalman::internal::SelfContainedWrapper<BaseObject, InternalizedParameters...>>
      : std::is_base_of<Eigen::MatrixBase<BaseObject>, BaseObject> {};
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
      detail::is_derived_eigen_matrix<std::decay_t<T>>::value;


  namespace detail
  {
    template<typename T>
    struct is_derived_eigen_array : std::false_type {};

    template<typename T, int Size>
    struct is_derived_eigen_array<Eigen::VectorBlock<T, Size>>
      : std::is_base_of<Eigen::ArrayBase<std::decay_t<T>>, std::decay_t<T>> {};

    template<typename BaseObject, typename...InternalizedParameters>
    struct is_derived_eigen_array<OpenKalman::internal::SelfContainedWrapper<BaseObject, InternalizedParameters...>>
      : std::is_base_of<Eigen::ArrayBase<BaseObject>, BaseObject> {};
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
      detail::is_derived_eigen_array<std::decay_t<T>>::value;


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
   * \note In Eigen, this does not need to be a \ref square_shaped.
   * \tparam NestedMatrix The nested matrix on which the identity is based.
   */
  template<typename NestedMatrix>
  using IdentityMatrix = Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<
    typename Eigen::internal::traits<std::decay_t<NestedMatrix>>::Scalar>, NestedMatrix>;


  namespace detail
  {
    template<typename T>
    struct is_eigen_wrapper : std::false_type {};

    template<typename N, typename L>
    struct is_eigen_wrapper<internal::LibraryWrapper<N, L>> : std::bool_constant<eigen_general<L, true>> {};
  } // namespace detail


  /**
   * \internal
   * \brief T is a \ref internal::LibraryWrapper "LibraryWrapper" for T based on the Eigen library.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_wrapper =
#else
  constexpr bool eigen_wrapper =
#endif
    detail::is_eigen_wrapper<std::decay_t<T>>::value;


  /**
   * \internal
   * \brief Alias for the Eigen version of LibraryWrapper.
   * \details A wrapper for OpenKalman classes so that they are treated exactly as native Eigen types.
   * \tparam NestedObject A non-Eigen class, for which an Eigen3 trait and evaluator is defined.
   */
#ifdef __cpp_concepts
  template<indexible NestedObject> requires (index_count_v<NestedObject> <= 2)
#else
  template<typename NestedObject>
#endif
  using EigenWrapper = internal::LibraryWrapper<NestedObject,
    std::conditional_t<eigen_array_general<NestedObject>,
      Eigen::Array<
        scalar_type_of_t<NestedObject>,
        dynamic_dimension<NestedObject, 0> ? Eigen::Dynamic : static_cast<int>(index_dimension_of_v<NestedObject, 0>),
        dynamic_dimension<NestedObject, 1> ? Eigen::Dynamic : static_cast<int>(index_dimension_of_v<NestedObject, 1>),
        layout_of_v<NestedObject> == Layout::right ? Eigen::RowMajor : Eigen::ColMajor>,
      Eigen::Matrix<
        scalar_type_of_t<NestedObject>,
        dynamic_dimension<NestedObject, 0> ? Eigen::Dynamic : static_cast<int>(index_dimension_of_v<NestedObject, 0>),
        dynamic_dimension<NestedObject, 1> ? Eigen::Dynamic : static_cast<int>(index_dimension_of_v<NestedObject, 1>),
        layout_of_v<NestedObject> == Layout::right ? Eigen::RowMajor : Eigen::ColMajor>>>;


  namespace detail
  {
    template<typename T>
    struct is_eigen_self_contained_wrapper : std::false_type {};

    template<typename N, typename...Ps>
    struct is_eigen_self_contained_wrapper<internal::SelfContainedWrapper<N, Ps...>> : std::bool_constant<eigen_general<N>> {};
  } // namespace detail


  /**
   * \internal
   * \brief T is an \ref internal::SelfContainedWrapper "SelfContainedWrapper" for T based on the Eigen library.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_self_contained_wrapper =
#else
  constexpr bool eigen_self_contained_wrapper =
#endif
    detail::is_eigen_self_contained_wrapper<std::decay_t<T>>::value;


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
