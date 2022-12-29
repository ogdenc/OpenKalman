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
 * \brief Forward declarations for OpenKalman's Eigen3 interface.
 */

#ifndef OPENKALMAN_EIGEN3_FORWARD_DECLARATIONS_HPP
#define OPENKALMAN_EIGEN3_FORWARD_DECLARATIONS_HPP

#include <type_traits>


/**
 * \namespace OpenKalman::Eigen3
 * \brief Namespace for all Eigen3-specific definitions.
 *
 * \internal
 * \namespace OpenKalman::Eigen3::internal
 * \brief Namespace for definitions internal to Eigen3-related functions.
 *
 * \namespace Eigen
 * \brief Eigen3's native namespace.
 *
 * \namespace Eigen::internal
 * \brief Eigen3's native namespace for internal definitions.
 */


namespace OpenKalman::Eigen3
{

  namespace internal
  {
    /**
     * \internal
     * \brief The ultimate Eigen base for OpenKalman classes.
     * \details This class is used mainly to distinguish OpenKalman classes from native Eigen classes which are
     * also derived from Eigen::MatrixBase or Eigen::ArrayBase.
     */
    struct Eigen3Base {};

    /**
     * \internal
     * \brief The ultimate base for Eigen-based adapter classes in OpenKalman.
     * \details This class adds base features required by Eigen.
     */
    template<typename Derived, typename NestedMatrix>
    struct Eigen3AdapterBase;
  } // namespace internal


  /**
   * \internal
   * \brief A dumb wrapper for OpenKalman classes so that they are treated exactly as native Eigen types.
   * \tparam T A non-Eigen class, for which an Eigen3 trait and evaluator is defined.
   */
  template<typename T>
  struct EigenWrapper;


  namespace detail
  {
    template<typename T>
    struct is_eigen_matrix_wrapper : std::false_type {};

    template<typename T>
    struct is_eigen_matrix_wrapper<EigenWrapper<T>>
      : std::is_base_of<Eigen::MatrixBase<std::decay_t<T>>, std::decay_t<T>> {};

    template<typename T, int Size>
    struct is_eigen_matrix_wrapper<Eigen::VectorBlock<T, Size>>
      : std::is_base_of<Eigen::MatrixBase<std::decay_t<T>>, std::decay_t<T>> {};
  }


  /**
   * \brief Specifies a native Eigen3 matrix or expression class deriving from Eigen::MatrixBase.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept native_eigen_matrix =
#else
  constexpr bool native_eigen_matrix =
#endif
    (std::is_base_of_v<Eigen::MatrixBase<std::decay_t<T>>, std::decay_t<T>> or
      detail::is_eigen_matrix_wrapper<std::decay_t<T>>::value) and
    (not std::is_base_of_v<internal::Eigen3Base, std::decay_t<T>>);


  namespace detail
  {
    template<typename T>
    struct is_eigen_array_wrapper : std::false_type {};

    template<typename T>
    struct is_eigen_array_wrapper<EigenWrapper<T>>
      : std::is_base_of<Eigen::ArrayBase<std::decay_t<T>>, std::decay_t<T>> {};

    template<typename T, int Size>
    struct is_eigen_array_wrapper<Eigen::VectorBlock<T, Size>>
      : std::is_base_of<Eigen::ArrayBase<std::decay_t<T>>, std::decay_t<T>> {};
  }


  /**
   * \brief Specifies a native Eigen3 array or expression class deriving from Eigen::ArrayBase.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept native_eigen_array =
#else
  constexpr bool native_eigen_array =
#endif
    (std::is_base_of_v<Eigen::ArrayBase<std::decay_t<T>>, std::decay_t<T>> or
      detail::is_eigen_array_wrapper<std::decay_t<T>>::value) and
    (not std::is_base_of_v<internal::Eigen3Base, std::decay_t<T>>);


  namespace detail
  {
    template<typename T>
    struct is_eigen_block : std::false_type {};

    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    struct is_eigen_block<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>> : std::true_type {};
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
    detail::is_eigen_block<std::decay_t<T>>::value;


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_convertible_to_native_eigen_matrix : std::false_type {};

    template<typename T>
    struct is_convertible_to_native_eigen_matrix<T, std::enable_if_t<
      std::is_constructible_v<Eigen::Matrix<
        typename Eigen::internal::traits<std::decay_t<T>>::Scalar,
        Eigen::internal::traits<std::decay_t<T>>::RowsAtCompileTime,
        Eigen::internal::traits<std::decay_t<T>>::ColsAtCompileTime>,
        const std::decay_t<T>&>>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies a native Eigen3 class that can be converted to Eigen::Matrix.
   * \details This should include any class in the Eigen library descending from Eigen::EigenBase.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept convertible_to_native_eigen_matrix =
    requires(const std::decay_t<T>& t) {
      Eigen::Matrix<
        typename Eigen::internal::traits<std::decay_t<T>>::Scalar,
        Eigen::internal::traits<std::decay_t<T>>::RowsAtCompileTime,
        Eigen::internal::traits<std::decay_t<T>>::ColsAtCompileTime> {t};
    };
#else
  constexpr bool convertible_to_native_eigen_matrix = detail::is_convertible_to_native_eigen_matrix<T>::value;
#endif


  /**
   * \brief An alias for the Eigen identity matrix.
   * \note In Eigen, this does not need to be a \ref square_matrix.
   * \tparam NestedMatrix The nested matrix on which the identity is based.
   */
  template<typename NestedMatrix>
  using IdentityMatrix =
    Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<
      typename Eigen::internal::traits<std::decay_t<NestedMatrix>>::Scalar>, NestedMatrix>;


  // ---------------- //
  //  eigen_matrix_t  //
  // ---------------- //

  namespace detail
  {
    template<std::size_t size>
    constexpr auto eigen_index_convert = size == dynamic_size ? Eigen::Dynamic : (Eigen::Index) size;
  }

  /**
   * \brief An alias for a self-contained, writable, native Eigen matrix.
   * \tparam Scalar Scalar type of the matrix (defaults to the Scalar type of T).
   * \tparam rows Number of rows in the native matrix (0 if not fixed at compile time).
   * \tparam cols Number of columns in the native matrix (0 if not fixed at compile time).
   */
  template<typename Scalar, std::size_t...dims>
  using eigen_matrix_t = std::conditional_t<sizeof...(dims) == 1,
    Eigen::Matrix<Scalar, detail::eigen_index_convert<dims>..., static_cast<Eigen::Index>(1)>,
    Eigen::Matrix<Scalar, detail::eigen_index_convert<dims>...>>;

} // namespace OpenKalman::Eigen3


#endif //OPENKALMAN_EIGEN3_FORWARD_DECLARATIONS_HPP
