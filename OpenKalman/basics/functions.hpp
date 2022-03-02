/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_FUNCTIONS_HPP
#define OPENKALMAN_FUNCTIONS_HPP

namespace OpenKalman
{
  using namespace interface;

  // ------------------------------- //
  //  Functions relating to indices  //
  // ------------------------------- //

  template<std::size_t N = 0, typename Arg>
  constexpr std::size_t runtime_dimension_of(Arg&& arg)
  {
    return interface::IndexTraits<std::decay_t<Arg>, N>::dimension_at_runtime(std::forward<Arg>(arg));
  }


  // --------------------------------- //
  //  make_dense_writable_matrix_from  //
  // --------------------------------- //

  /**
   * \brief Convert the argument to a dense, writable matrix of a type based on native matrix T.
   */
#ifdef __cpp_concepts
  template<typename Arg> requires
    requires(Arg&& arg) { EquivalentDenseWritableMatrix<std::decay_t<Arg>>::convert(std::forward<Arg>(arg)); }
#else
  template<typename Arg,
    typename = std::void_t<decltype(EquivalentDenseWritableMatrix<std::decay_t<Arg>>::template convert<Arg&&>)>>
#endif
  constexpr decltype(auto)
  make_dense_writable_matrix_from(Arg&& arg) noexcept
  {
    using Trait = EquivalentDenseWritableMatrix<std::decay_t<Arg>>;
    using Nat = std::remove_reference_t<decltype(Trait::convert(std::declval<Arg&&>()))>;

    static_assert(not std::is_const_v<Nat>, "EquivalentDenseWritableMatrix::convert logic error: returns const result");

    if constexpr (std::is_same_v<std::decay_t<Arg>, Nat>)
      return std::forward<Arg>(arg);
    else
      return Trait::convert(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Create a dense, writable matrix with size and shape based on M, filled with a set of scalar components
   * \tparam M The matrix or array on which the new matrix is patterned.
   * \tparam rows An optional row dimension for the new matrix. By default, M's row dimension is used.
   * \tparam columns An optional column dimension for the new matrix. By default, M's column dimension is used.
   * \tparam Scalar An optional scalar type for the new matrix. By default, M's scalar type is used.
   * \param args Scalar values to fill the new matrix.
   */
#ifdef __cpp_concepts
  template<indexible M, std::size_t rows = row_dimension_of_v<M>, std::size_t columns = column_dimension_of_v<M>,
      typename Scalar = scalar_type_of_t<M>, std::convertible_to<const Scalar> ... Args>
  requires (sizeof...(Args) > 0) and
    ((rows == dynamic_size and columns == dynamic_size) or
     (rows != dynamic_size and columns != dynamic_size and sizeof...(Args) == rows * columns) or
     (columns == dynamic_size and sizeof...(Args) % rows == 0) or
     (rows == dynamic_size and sizeof...(Args) % columns == 0)) and
    requires { typename MatrixTraits<std::decay_t<decltype(
      interface::EquivalentDenseWritableMatrix<std::decay_t<M>, rows, columns, Scalar>::convert(std::declval<M>()))>>; }
#else
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wdiv-by-zero"
  template<typename M, std::size_t rows = row_dimension_of_v<M>, std::size_t columns = column_dimension_of_v<M>,
      typename Scalar = scalar_type_of_t<M>, typename ... Args, std::enable_if_t<indexible<M> and
    (sizeof...(Args) > 0) and (std::is_convertible_v<Args, const Scalar> and ...) and
    ((rows == dynamic_size and columns == dynamic_size) or
     (rows != dynamic_size and columns != dynamic_size and sizeof...(Args) == rows * columns) or
     (columns == dynamic_size and sizeof...(Args) % rows == 0) or
     (rows == dynamic_size and sizeof...(Args) % columns == 0)) and
    std::is_void<std::void_t<MatrixTraits<std::decay_t<decltype(
      interface::EquivalentDenseWritableMatrix<std::decay_t<M>, rows, columns, Scalar>::convert(std::declval<M>()))>>>>::value, int> = 0>
#endif
  inline auto
  make_dense_writable_matrix_from(const Args ... args)
  {
    using Trait = EquivalentDenseWritableMatrix<std::decay_t<M>, rows, columns, Scalar>;
    using Nat = std::decay_t<decltype(Trait::convert(std::declval<M>()))>;
    return MatrixTraits<Nat>::make(static_cast<const Scalar>(args)...);
  }
#ifndef __cpp_concepts
# pragma GCC diagnostic pop
#endif


  // ----------------------------------------- //
  //  make_default_dense_writable_matrix_like  //
  // ----------------------------------------- //

  /**
   * \brief Create a dense, writable matrix with size and shape modeled at least partially on T
   * \tparam T The matrix or array on which the new matrix is patterned.
   * \tparam rows An optional row dimension for the new matrix. By default, T's row dimension is used.
   * \tparam columns An optional column dimension for the new matrix. By default, T's column dimension is used.
   * \tparam Scalar An optional scalar type for the new matrix. By default, T's scalar type is used.
   * \param e Any necessary runtime dimensions.
   */
#ifdef __cpp_concepts
  template<indexible T, std::size_t rows = row_dimension_of_v<T>, std::size_t columns = column_dimension_of_v<T>,
    typename Scalar = scalar_type_of_t<T>, std::convertible_to<std::size_t> ... runtime_dimensions>
  requires (sizeof...(runtime_dimensions) == (rows == dynamic_size ? 1 : 0) + (columns == dynamic_size ? 1 : 0)) and
    requires(runtime_dimensions...e) {
      EquivalentDenseWritableMatrix<std::decay_t<T>, rows, columns, Scalar>::make_default(e...); }
#else
  template<typename T, std::size_t rows = row_dimension_of_v<T>, std::size_t columns = column_dimension_of_v<T>,
    typename Scalar = scalar_type_of_t<T>, typename ... runtime_dimensions, std::enable_if_t<indexible<T> and
    (sizeof...(runtime_dimensions) == (rows == dynamic_size ? 1 : 0) + (columns == dynamic_size ? 1 : 0)) and
    (std::is_convertible_v<runtime_dimensions, std::size_t> and ...) and
    std::is_void<std::void_t<decltype(EquivalentDenseWritableMatrix<std::decay_t<T>, rows, columns, Scalar>::make_default(
      std::declval<runtime_dimensions>()...))>>::value, int> = 0>
#endif
  inline auto
  make_default_dense_writable_matrix_like(runtime_dimensions...e)
  {
    return EquivalentDenseWritableMatrix<std::decay_t<T>, rows, columns, Scalar>::make_default(e...);
  }


  /**
   * \overload
   * \brief Make a default, dense, writable matrix based on the argument, but specifying new dimensions.
   * \tparam rows The new number of rows (may be \ref dynamic_shape)
   * \tparam columns The new number of columns (may be \ref dynamic_shape)
   * \tparam Scalar An optional scalar type for the new matrix. By default, T's scalar type is used.
   */
#ifdef __cpp_concepts
  template<indexible T, std::size_t rows = index_dimension_of_v<T, 0>, std::size_t columns = index_dimension_of_v<T, 1>,
    typename Scalar = scalar_type_of_t<T>>
#else
  template<typename T, std::size_t rows = index_dimension_of_v<T, 0>, std::size_t columns = index_dimension_of_v<T, 1>,
    typename Scalar = scalar_type_of_t<T>, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr auto
  make_default_dense_writable_matrix_like(const T& t)
  {
    if constexpr (rows == dynamic_size and columns == dynamic_size)
      return make_default_dense_writable_matrix_like<T, rows, columns, Scalar>(runtime_dimension_of<0>(t), runtime_dimension_of<1>(t));
    else if constexpr (rows == dynamic_size)
      return make_default_dense_writable_matrix_like<T, rows, columns, Scalar>(runtime_dimension_of<0>(t));
    else if constexpr (columns == dynamic_size)
      return make_default_dense_writable_matrix_like<T, rows, columns, Scalar>(runtime_dimension_of<1>(t));
    else
      return make_default_dense_writable_matrix_like<T, rows, columns, Scalar>();
  }


  /**
   * \overload
   * \brief Make a default, dense, writable matrix based on the argument, but specifying new dimensions.
   * \tparam dims One or more dimensions associated with indices of the new matrix (may be \ref dynamic_shape)
   * \tparam Scalar An optional scalar value for the new matrix.
   * \tparam T The matrix or array on which the new matrix is patterned.
   */
#ifdef __cpp_concepts
  template<std::size_t...dims, typename...Scalar, indexible T> requires
    (sizeof...(dims) > 0) and (sizeof...(dims) <= 2) and (sizeof...(Scalar) <= 1)
#else
  template<std::size_t...dims, typename...Scalar, typename T, std::enable_if_t<
    indexible<T> and (sizeof...(dims) > 0) and (sizeof...(dims) <= 2) and (sizeof...(Scalar) <= 1), int> = 0>
#endif
  constexpr decltype(auto)
  make_default_dense_writable_matrix_like(T&& t)
  {
    return make_default_dense_writable_matrix_like<T, dims..., Scalar...>(std::forward<T>(t));
  }


  // ------------------ //
  //  to_native_matrix  //
  // ------------------ //

  /**
   * \brief If it isn't already, convert Arg to a native matrix in library T.
   * \details The new matix will be one in which basic matrix operations are defined.
   * \tparam T A matrix from the library to which Arg is to be converted.
   * \tparam Arg The argument
   */
#ifdef __cpp_concepts
  template<indexible T, indexible Arg>
#else
  template<typename T, typename Arg, std::enable_if_t<indexible<T> and indexible<Arg>, int> = 0>
#endif
  inline auto
  to_native_matrix(Arg&& arg)
  {
    return EquivalentDenseWritableMatrix<std::decay_t<T>>::to_native_matrix(std::forward<Arg>(arg));
  }


  // ----------------------- //
  //  make_zero_matrix_like  //
  // ----------------------- //

  /**
   * \brief Make a zero matrix with size and shape modeled at least partially on T
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   * \tparam rows An optional row dimension for the new zero matrix. By default, T's row dimension is used.
   * \tparam columns An optional column dimension for the new zero matrix. By default, T's column dimension is used.
   * \tparam Scalar An optional scalar type for the new zero matrix. By default, T's scalar type is used.
   * \param e Any necessary runtime dimensions.
   */
#ifdef __cpp_concepts
  template<indexible T, std::size_t rows = row_dimension_of_v<T>, std::size_t columns = column_dimension_of_v<T>,
    typename Scalar = scalar_type_of_t<T>, std::convertible_to<std::size_t>...runtime_dimensions>
  requires (sizeof...(runtime_dimensions) == (rows == dynamic_size ? 1 : 0) + (columns == dynamic_size ? 1 : 0))
#else
  template<typename T, std::size_t rows = row_dimension_of<T>::value, std::size_t columns = column_dimension_of<T>::value,
    typename Scalar = typename scalar_type_of<T>::type, typename...runtime_dimensions, std::enable_if_t<
      indexible<T> and
      (sizeof...(runtime_dimensions) == (rows == dynamic_size ? 1 : 0) + (columns == dynamic_size ? 1 : 0)) and
      (std::is_convertible_v<runtime_dimensions, std::size_t> and ...), int> = 0>
#endif
  constexpr auto
  make_zero_matrix_like(runtime_dimensions...e)
  {
    return SingleConstantMatrixTraits<std::decay_t<T>, rows, columns, Scalar>::make_zero_matrix(e...);
  }


  /**
   * \overload
   * \brief Make a zero matrix based on the argument, but specifying new dimensions.
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   * \tparam rows The new number of rows (may be \ref dynamic_shape)
   * \tparam columns The new number of columns (may be \ref dynamic_shape)
   */
#ifdef __cpp_concepts
  template<indexible T, std::size_t rows = row_dimension_of_v<T>, std::size_t columns = column_dimension_of_v<T>,
    typename Scalar = scalar_type_of_t<T>>
#else
  template<typename T, std::size_t rows = row_dimension_of_v<T>, std::size_t columns = column_dimension_of_v<T>,
    typename Scalar = scalar_type_of_t<T>, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr decltype(auto)
  make_zero_matrix_like(T&& t)
  {
    if constexpr (zero_matrix<T> and rows == row_dimension_of_v<T> and columns == column_dimension_of_v<T> and
        std::is_same_v<Scalar, scalar_type_of_t<T>>)
      return std::forward<T>(t);
    else if constexpr (rows == dynamic_size and columns == dynamic_size)
      return make_zero_matrix_like<T, rows, columns, Scalar>(runtime_dimension_of<0>(t), runtime_dimension_of<1>(t));
    else if constexpr (rows == dynamic_size)
      return make_zero_matrix_like<T, rows, columns, Scalar>(runtime_dimension_of<0>(t));
    else if constexpr (columns == dynamic_size)
      return make_zero_matrix_like<T, rows, columns, Scalar>(runtime_dimension_of<1>(t));
    else
      return make_zero_matrix_like<T, rows, columns, Scalar>();
  }


  /**
   * \overload
   * \brief Make a zero matrix based on the argument, but specifying new dimensions.
   * \tparam dims One or more dimensions associated with indices of the new matrix (may be \ref dynamic_shape)
   * \tparam Scalar An optional scalar value for the new matrix.
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   */
#ifdef __cpp_concepts
  template<std::size_t...dims, typename...Scalar, indexible T> requires
    (sizeof...(dims) > 0) and (sizeof...(dims) <= 2) and (sizeof...(Scalar) <= 1)
#else
  template<std::size_t...dims, typename...Scalar, typename T, std::enable_if_t<
    indexible<T> and (sizeof...(dims) > 0) and (sizeof...(dims) <= 2) and (sizeof...(Scalar) <= 1), int> = 0>
#endif
  constexpr decltype(auto)
  make_zero_matrix_like(T&& t)
  {
    return make_zero_matrix_like<T, dims..., Scalar...>(std::forward<T>(t));
  }


  // --------------------------- //
  //  make_constant_matrix_like  //
  // --------------------------- //

  /**
   * \brief Make a single-constant matrix with size and shape modeled at least partially on T
   * \tparam T The matrix or array on which the new matrix is patterned.
   * \tparam constant The constant.
   * \tparam rows An optional row dimension for the new matrix. By default, T's row dimension is used.
   * \tparam columns An optional column dimension for the new matrix. By default, T's column dimension is used.
   * \tparam Scalar An optional scalar type for the new matrix. By default, T's scalar type is used.
   * \param e Any necessary runtime dimensions.
   */
#ifdef __cpp_concepts
  template<indexible T, auto constant, std::size_t rows = row_dimension_of_v<T>,
    std::size_t columns = column_dimension_of_v<T>, typename Scalar = scalar_type_of_t<T>,
    std::convertible_to<std::size_t>...runtime_dimensions>
  requires (sizeof...(runtime_dimensions) == (rows == dynamic_size ? 1 : 0) + (columns == dynamic_size ? 1 : 0))
#else
  template<typename T, auto constant, std::size_t rows = row_dimension_of<T>::value,
    std::size_t columns = column_dimension_of<T>::value, typename Scalar = typename scalar_type_of<T>::type,
    typename...runtime_dimensions, std::enable_if_t<indexible<T> and
      (sizeof...(runtime_dimensions) == (rows == dynamic_size ? 1 : 0) + (columns == dynamic_size ? 1 : 0)) and
      (std::is_convertible_v<runtime_dimensions, std::size_t> and ...), int> = 0>
#endif
  constexpr auto
  make_constant_matrix_like(runtime_dimensions...e)
  {
    return SingleConstantMatrixTraits<std::decay_t<T>, rows, columns, Scalar>::template make_constant_matrix<constant>(e...);
  }


  /**
   * \overload
   * \brief Make a single-constant matrix based on the argument, but specifying new dimensions.
   * \tparam T The matrix or array on which the new matrix is patterned.
   * \tparam constant The constant.
   * \tparam rows The new number of rows (may be \ref dynamic_shape)
   * \tparam columns The new number of columns (may be \ref dynamic_shape)
   * \tparam Scalar An optional scalar type for the new matrix. By default, T's scalar type is used.
   */
#ifdef __cpp_concepts
  template<indexible T, auto constant, std::size_t rows = row_dimension_of_v<T>,
    std::size_t columns = column_dimension_of_v<T>, typename Scalar = scalar_type_of_t<T>>
#else
  template<typename T, auto constant, std::size_t rows = row_dimension_of_v<T>,
    std::size_t columns = column_dimension_of_v<T>, typename Scalar = scalar_type_of_t<T>,
    std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr decltype(auto)
  make_constant_matrix_like(T&& t)
  {
    constexpr bool constants_match = []{
      if constexpr (constant_matrix<T>) return are_within_tolerance(constant, constant_coefficient_v<T>);
      else return false;
    }();

    if constexpr (constants_match and row_dimension_of_v<T> == rows and column_dimension_of_v<T> == columns and
        std::is_same_v<scalar_type_of_t<T>, Scalar>)
      return std::forward<T>(t);
    else if constexpr (rows == dynamic_size and columns == dynamic_size)
      return make_constant_matrix_like<T, constant, rows, columns, Scalar>(runtime_dimension_of<0>(t), runtime_dimension_of<1>(t));
    else if constexpr (rows == dynamic_size)
      return make_constant_matrix_like<T, constant, rows, columns, Scalar>(runtime_dimension_of<0>(t));
    else if constexpr (columns == dynamic_size)
      return make_constant_matrix_like<T, constant, rows, columns, Scalar>(runtime_dimension_of<1>(t));
    else
      return make_constant_matrix_like<T, constant, rows, columns, Scalar>();
  }


  /**
   * \overload
   * \brief Make a single-constant matrix based on the argument, but specifying new dimensions.
   * \tparam dims Zero or more dimensions associated with indices of the new matrix (may be \ref dynamic_shape)
   * \tparam Scalar An optional scalar value for the new matrix.
   * \tparam T The matrix or array on which the new matrix is patterned.
   */
#ifdef __cpp_concepts
  template<auto constant, std::size_t...dims, typename...Scalar, indexible T> requires
    (sizeof...(dims) <= 2) and (sizeof...(Scalar) <= 1)
#else
  template<auto constant, std::size_t...dims, typename...Scalar, typename T, std::enable_if_t<
    indexible<T> and (sizeof...(dims) <= 2) and (sizeof...(Scalar) <= 1), int> = 0>
#endif
  constexpr decltype(auto)
  make_constant_matrix_like(T&& t)
  {
    return make_constant_matrix_like<T, constant, dims..., Scalar...>(std::forward<T>(t));
  }


  // --------------------------- //
  //  make_identity_matrix_like  //
  // --------------------------- //

  /**
   * \brief Make an identity matrix with a size and shape modeled at least partially on T.
   * \tparam T The matrix or array on which the identity matrix is patterned.
   * \tparam dimension An optional row and column dimension for the new zero matrix. By default, the function uses
   * T's row dimension, if it is not dynamic, or otherwise T's column dimension.
   * \tparam Scalar An optional scalar type for the new zero matrix. By default, T's scalar type is used.
   * \param e Any necessary runtime dimensions.
   */
#ifdef __cpp_concepts
  template<indexible T, std::size_t dimension = dynamic_rows<T> ? column_dimension_of_v<T> : row_dimension_of_v<T>,
      typename Scalar = scalar_type_of_t<T>, std::convertible_to<std::size_t>...runtime_dimensions> requires
    (sizeof...(runtime_dimensions) == (dimension == dynamic_size ? 1 : 0))
#else
  template<typename T, std::size_t dimension = dynamic_rows<T> ? column_dimension_of<T>::value : row_dimension_of<T>::value,
    typename Scalar = typename scalar_type_of<T>::type, typename...runtime_dimensions, std::enable_if_t<indexible<T> and
      (sizeof...(runtime_dimensions) == (dimension == dynamic_size ? 1 : 0)) and
      (std::is_convertible_v<runtime_dimensions, std::size_t> and ...), int> = 0>
#endif
  constexpr auto
  make_identity_matrix_like(runtime_dimensions...e)
  {
    return SingleConstantDiagonalMatrixTraits<std::decay_t<T>, dimension, Scalar>::make_identity_matrix(e...);
  }


  /**
   * \overload
   * \brief Make an identity matrix based on a matrix argument, but with a specified dimension.
   * \tparam dimension The new dimension of the identity matrix (may be \ref dynamic_size)
   */
#ifdef __cpp_concepts
  template<indexible T, std::size_t dimension, typename Scalar = scalar_type_of_t<T>>
#else
  template<typename T, std::size_t dimension, typename Scalar = scalar_type_of_t<T>,
    std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr decltype(auto)
  make_identity_matrix_like(T&& t)
  {
    if constexpr (identity_matrix<T> and dimension == index_dimension_of_v<T, 0>)
    {
      return std::forward<T>(t);
    }
    else if constexpr (dimension == dynamic_size)
    {
      assert(runtime_dimension_of<0>(t) == runtime_dimension_of<1>(t));
      return make_identity_matrix_like<T, dimension, Scalar>(runtime_dimension_of<0>(t));
    }
    else
    {
      return make_identity_matrix_like<T, dimension, Scalar>();
    }
  }


  /**
   * \overload
   * \brief Make an identity matrix shape and size as the argument.
   * \note The argument must be a square matrix.
   */
#ifdef __cpp_concepts
  template<typename T> requires any_dynamic_dimension<T> or square_matrix<T>
#else
  template<typename T, std::enable_if_t<any_dynamic_dimension<T> or square_matrix<T>, int> = 0>
#endif
  constexpr decltype(auto)
  make_identity_matrix_like(T&& t)
  {
    if constexpr (any_dynamic_dimension<T>) assert(runtime_dimension_of<0>(t) == runtime_dimension_of<1>(t));

    if constexpr (identity_matrix<T>)
      return std::forward<T>(t);
    else
      return make_identity_matrix_like<T, index_dimension_of_v<T, 0>>(std::forward<T>(t));
  }


  /**
   * \overload
   * \brief Make an identity matrix based on the argument, but specifying new dimensions.
   * \tparam dims The dimension associated with indices of the new matrix (may be \ref dynamic_shape)
   * \tparam Scalar An optional scalar value for the new matrix.
   * \tparam T The matrix or array on which the new matrix is patterned.
   */
#ifdef __cpp_concepts
  template<std::size_t dimension, typename...Scalar, indexible T> requires (sizeof...(Scalar) <= 1)
#else
  template<std::size_t dimension, typename...Scalar, typename T, std::enable_if_t<
    indexible<T> and (sizeof...(Scalar) <= 1), int> = 0>
#endif
  constexpr decltype(auto)
  make_identity_matrix_like(T&& t)
  {
    return make_identity_matrix_like<T, dimension, Scalar...>(std::forward<T>(t));
  }


  // --------------- //
  //  nested_matrix  //
  // --------------- //

  /**
   * \brief Retrieve a nested matrix of Arg, if it exists.
   * \tparam i Index of the nested matrix (0 for the 1st, 1 for the 2nd, etc.).
   * \tparam Arg A wrapper that has at least one nested matrix.
   * \internal \sa interface::Dependencies::get_nested_matrix
   */
#ifdef __cpp_concepts
  template<std::size_t i = 0, typename Arg> requires
    (i < std::tuple_size_v<typename Dependencies<std::decay_t<Arg>>::type>) and
    requires(Arg&& arg) { Dependencies<std::decay_t<Arg>>::template get_nested_matrix<i>(std::forward<Arg>(arg)); }
#else
  template<std::size_t i = 0, typename Arg,
    std::enable_if_t<(i < std::tuple_size<typename Dependencies<std::decay_t<Arg>>::type>::value), int> = 0,
    typename = std::void_t<decltype(Dependencies<std::decay_t<Arg>>::template get_nested_matrix<i>(std::declval<Arg&&>()))>>
#endif
  constexpr decltype(auto)
  nested_matrix(Arg&& arg) noexcept
  {
      return Dependencies<std::decay_t<Arg>>::template get_nested_matrix<i>(std::forward<Arg>(arg));
  }


  // --------------------- //
  //  make_self_contained  //
  // --------------------- //

  namespace detail
  {
    template<typename Tup, std::size_t...I>
    constexpr bool all_lvalue_ref_dependencies_impl(std::index_sequence<I...>)
    {
      return ((sizeof...(I) > 0) and ... and std::is_lvalue_reference_v<std::tuple_element_t<I, Tup>>);
    }

#ifdef __cpp_concepts
    template<typename T>
    concept all_lvalue_ref_dependencies = (not Dependencies<std::decay_t<T>>::has_runtime_parameters) and
      all_lvalue_ref_dependencies_impl<typename Dependencies<std::decay_t<T>>::type>(
        std::make_index_sequence<std::tuple_size_v<typename Dependencies<std::decay_t<T>>::type>> {});
#else
    template<typename T, typename = void>
    struct has_no_runtime_parameters_impl : std::false_type {};

    template<typename T>
    struct has_no_runtime_parameters_impl<T, std::enable_if_t<not Dependencies<T>::has_runtime_parameters>>
      : std::true_type {};


    template<typename T, typename = void>
    struct all_lvalue_ref_dependencies_detail : std::false_type {};

    template<typename T>
    struct all_lvalue_ref_dependencies_detail<T, std::void_t<typename Dependencies<T>::type>>
      : std::bool_constant<has_no_runtime_parameters_impl<T>::value and
        (all_lvalue_ref_dependencies_impl<typename Dependencies<T>::type>(
          std::make_index_sequence<std::tuple_size_v<typename Dependencies<T>::type>> {}))> {};

    template<typename T>
    constexpr bool all_lvalue_ref_dependencies = all_lvalue_ref_dependencies_detail<std::decay_t<T>>::value;


    template<typename T, typename = void>
    struct convert_to_self_contained_is_defined : std::false_type {};

    template<typename T>
    struct convert_to_self_contained_is_defined<T,
      std::void_t<decltype(Dependencies<std::decay_t<T>>::convert_to_self_contained(std::declval<T&&>()))>>
      : std::true_type {};
#endif
  } // namespace detail


  /**
   * \brief Convert to a self-contained version of Arg that can be returned in a function.
   * \details If any types Ts are included, Arg will not be converted to a self-contained version if every Ts is either
   * an lvalue reference or has a nested matrix that is an lvalue reference. This is to allow a function, taking Ts...
   * as lvalue-reference inputs or as rvalue-reference inputs that nest lvalue-references to other matrices, to avoid
   * unnecessary conversion because the referenced objects are accessible outside the scope of the function and do not
   * result in dangling references.
   * The following example adds two matrices arg1 and arg2 together and returns a self-contained matrix, unless
   * <em>both</em> Arg1 and Arg2 are lvalue references or their nested matrices are lvalue references, in which case
   * the result of the addition is returned without eager evaluation:
   * \code
   *   template<typename Arg1, typename Arg2>
   *   static decltype(auto) add(Arg1&& arg1, Arg2&& arg2)
   *   {
   *     return make_self_contained<Arg1, Arg2>(arg1 + arg2);
   *   }
   * \endcode
   * \tparam Ts Generally, these will be forwarding-reference arguments to the directly enclosing function. If all of
   * Ts... are lvalue references, Arg is returned without modification (i.e., without any potential eager evaluation).
   * \tparam Arg The potentially non-self-contained argument to be converted
   * \return A self-contained version of Arg (if it is not already self-contained)
   * \internal \sa interface::Dependencies
   */
  template<typename ... Ts, typename Arg>
  constexpr decltype(auto)
  make_self_contained(Arg&& arg) noexcept
  {
    if constexpr (self_contained<Arg> or
      ((sizeof...(Ts) > 0) and ... and (std::is_lvalue_reference_v<Ts> or detail::all_lvalue_ref_dependencies<Ts>)))
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (std::is_lvalue_reference_v<Arg> and self_contained<std::decay_t<Arg>> and
      std::is_copy_constructible_v<std::decay_t<Arg>>)
    {
      // If it is not self-contained because it is an lvalue reference, simply return a copy.
      return std::decay_t<Arg> {arg};
    }
#ifdef __cpp_concepts
    else if constexpr (requires {Dependencies<std::decay_t<Arg>>::convert_to_self_contained(std::forward<Arg>(arg)); })
#else
    else if constexpr (detail::convert_to_self_contained_is_defined<Arg>::value)
#endif
    {
      return Dependencies<std::decay_t<Arg>>::convert_to_self_contained(std::forward<Arg>(arg));
    }
    else
    {
      return make_dense_writable_matrix_from(std::forward<Arg>(arg));
    }
  }


  // =================== //
  //  Element functions  //
  // =================== //

  namespace detail
  {
    template<bool set, typename Arg, std::size_t...seq, typename...I>
    inline void check_index_bounds(const Arg& arg, std::index_sequence<seq...>, I...i)
    {
      if constexpr (sizeof...(I) == 1)
      {
        auto c = runtime_dimension_of<1>(arg);
        if (c == 1)
        {
          auto r = runtime_dimension_of<0>(arg);
          if ((static_cast<std::size_t>(i),...) >= r)
            throw std::out_of_range {((std::string {set ? "s" : "g"} + "et_element:") + " Row index (which is " +
            std::to_string((i,...)) + ") is not in range 0 <= i < " + std::to_string(r) + ".")};
        }
        else
        {
          if ((static_cast<std::size_t>(i),...) >= c)
            throw std::out_of_range {((std::string {set ? "s" : "g"} + "et_element:") + " Column index (which is " +
            std::to_string((i,...)) + ") is not in range 0 <= i < " + std::to_string(c) + ".")};
        }
      }
      else
      {
        (((static_cast<std::size_t>(i) >= runtime_dimension_of<seq>(arg)) ?
          throw std::out_of_range {((std::string {set ? "s" : "g"} + "et_element:") + ... +
            (" Index " + std::to_string(seq) + " (which is " + std::to_string(i) + ") is not in range 0 <= i < " +
            std::to_string(runtime_dimension_of<seq>(arg)) + "."))} :
          false) , ...);
      }
    }
  }


  /// Get element of matrix arg using I... indices.
#ifdef __cpp_concepts
  template<typename Arg, std::convertible_to<std::size_t>...I> requires
    element_gettable<Arg, std::conditional_t<std::same_as<I, std::size_t>, I, std::size_t>...>
  constexpr auto get_element(Arg&& arg, const I...i)
  {
    detail::check_index_bounds<false>(arg, std::make_index_sequence<sizeof...(I)>{}, i...);
    return interface::GetElement<std::decay_t<Arg>, I...>::get(std::forward<Arg>(arg), i...);
  }
#else
  template<typename Arg, typename...I, std::enable_if_t<(std::is_convertible_v<I, std::size_t> and ...) and
    element_gettable<Arg, std::conditional_t<std::is_same_v<I, std::size_t>, I, std::size_t>...> and
    (sizeof...(I) != 1 or column_vector<Arg> or row_vector<Arg>), int> = 0>
  constexpr auto get_element(Arg&& arg, const I...i)
  {
    detail::check_index_bounds<false>(arg, std::make_index_sequence<sizeof...(I)>{}, i...);
    return interface::GetElement<std::decay_t<Arg>, void, I...>::get(std::forward<Arg>(arg), i...);
  }
#endif


  /// Set element to s using I... indices.
#ifdef __cpp_concepts
  template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>&> Scalar, std::convertible_to<std::size_t>...I>
    requires element_settable<Arg&, std::conditional_t<std::same_as<I, std::size_t>, I, std::size_t>...>
  inline void set_element(Arg& arg, Scalar s, const I...i)
  {
    detail::check_index_bounds<true>(arg, std::make_index_sequence<sizeof...(I)>{}, i...);
    return interface::SetElement<std::decay_t<Arg>, I...>::set(arg, s, i...);
  }
#else
  template<typename Arg, typename Scalar, typename...I, std::enable_if_t<
    (std::is_convertible_v<I, std::size_t> and ...) and
    std::is_convertible_v<Scalar, const scalar_type_of_t<Arg>&> and
    element_settable<Arg&, std::conditional_t<std::is_same_v<I, std::size_t>, I, std::size_t>...>, int> = 0>
  inline void set_element(Arg& arg, Scalar s, const I...i)
  {
    detail::check_index_bounds<true>(arg, std::make_index_sequence<sizeof...(I)>{}, i...);
    return interface::SetElement<std::decay_t<Arg>, void, I...>::set(arg, s, i...);
  }
#endif


  // ========================= //
  //  Element-wise operations  //
  // ========================= //

  /**
   * \brief Fold an operation across the elements of a matrix
   * \detail BinaryFunction must be invocable with two values, the first an accumulator and the second of type
   * <code>scalar_type_of_t<Arg></code>. It returns the accumulated value. After each iteration, the result of the
   * operation is used as the accumulator for the next iteration.
   * \tparam BinaryFunction A binary function (e.g. std::plus, std::multiplies)
   * \tparam Accum An accumulator
   * \tparam order The element order over which to perform the operation
   */
#ifdef __cpp_concepts
  template<ElementOrder order = ElementOrder::column_major, typename BinaryFunction, typename Accum, typename Arg>
    requires std::is_invocable_r_v<const std::remove_reference_t<Accum>&, const BinaryFunction&,
      Accum&&, scalar_type_of_t<Arg>> and std::move_constructible<std::decay_t<Accum>> and
    std::copy_constructible<std::decay_t<Accum>>
  constexpr decltype(auto) fold(const BinaryFunction& b, Accum&& accum, Arg&& arg)
#else
  template<ElementOrder order = ElementOrder::column_major, typename BinaryFunction, typename Accum, typename Arg,
    std::enable_if_t<std::is_invocable<const std::remove_reference_t<Accum>&, const BinaryFunction&,
      Accum&&, scalar_type_of_t<Arg>>::value and std::is_move_constructible<std::decay_t<Accum>>::value and
    std::is_copy_constructible<std::decay_t<Accum>>::value, int> = 0>
  constexpr decltype(auto) fold(const BinaryFunction& b, Accum&& accum, Arg&& arg)
#endif
  {
    using Scalar = scalar_type_of_t<Arg>;

    if constexpr ((constant_matrix<Arg> or diagonal_matrix<Arg>) and
      (std::is_same_v<BinaryFunction, std::plus<void>> or std::is_same_v<BinaryFunction, std::plus<Scalar>>))
    {
      if constexpr (zero_matrix<Arg>)
        return Scalar(0);
      else if constexpr (constant_matrix<Arg>)
        return Scalar(constant_coefficient_v<Arg> * runtime_dimension_of<0>(arg) * runtime_dimension_of<1>(arg));
      else if constexpr (constant_diagonal_matrix<Arg>)
        return Scalar(constant_diagonal_coefficient_v<Arg> * runtime_dimension_of<0>(arg));
      else
      {
        static_assert(diagonal_matrix<Arg>);
        return OpenKalman::fold(b, std::forward<Accum>(accum), diagonal_of(std::forward<Arg>(arg)));
      }
    }
    else if constexpr ((constant_matrix<Arg> or triangular_matrix<Arg>) and
      (std::is_same_v<BinaryFunction, std::multiplies<void>> or std::is_same_v<BinaryFunction, std::multiplies<Scalar>>))
    {
      if constexpr (zero_matrix<Arg> or triangular_matrix<Arg>)
        return Scalar(0);
      else
      {
        static_assert(constant_matrix<Arg>);
        if constexpr (any_dynamic_dimension<Arg>)
          return Scalar(std::pow(constant_coefficient_v<Arg>, runtime_dimension_of<0>(arg) * runtime_dimension_of<1>(arg)));
        else
          return internal::constexpr_pow(constant_coefficient_v<Arg>, row_dimension_of_v<Arg> * column_dimension_of_v<Arg>);
      }
    }
    else
    {
      return interface::ElementWiseOperations<std::decay_t<Arg>>::template fold<order>(
        b, std::forward<Accum>(accum), std::forward<Arg>(arg));
    }
  }


  // ============= //
  //  Conversions  //
  // ============= //

  /**
   * \brief Convert a column vector into a diagonal matrix.
   * \tparam Arg A column vector matrix
   * \returns A diagonal matrix
   */
#ifdef __cpp_concepts
  template<typename Arg> requires column_vector<Arg> or dynamic_columns<Arg>
#else
  template<typename Arg, std::enable_if_t<column_vector<Arg> or dynamic_columns<Arg>, int> = 0>
#endif
  inline decltype(auto)
  to_diagonal(Arg&& arg)
  {
    constexpr auto dim = row_dimension_of_v<Arg>;

    if constexpr (dim == 1)
    {
      if constexpr (dynamic_columns<Arg>) if (runtime_dimension_of<1>(arg) != 1) throw std::invalid_argument {
        "Argument of to_diagonal must be a column vector, not a row vector"};
      return std::forward<Arg>(arg);
    }
    else if constexpr (zero_matrix<Arg> and dim != dynamic_size)
    {
      // note, the interface function should deal with a zero matrix of uncertain size.

      if constexpr (dynamic_columns<Arg>) if (runtime_dimension_of<1>(arg) != 1) throw std::invalid_argument {
        "Argument of to_diagonal must have 1 column; instead it has " + std::to_string(runtime_dimension_of<1>(arg))};
      return make_zero_matrix_like<Arg, dim, dim>();
    }
    else
    {
      return interface::Conversions<std::decay_t<Arg>>::to_diagonal(std::forward<Arg>(arg));
    }
  }


  namespace detail
  {
    template<typename Arg>
    inline void check_if_square_at_runtime(const Arg& arg)
    {
      if (runtime_dimension_of<0>(arg) != runtime_dimension_of<1>(arg))
        throw std::invalid_argument {"Argument of diagonal_of must be a square matrix; instead it has " +
          std::to_string(runtime_dimension_of<0>(arg)) + " rows and " + std::to_string(runtime_dimension_of<1>(arg)) +
          "columns"};
    };
  }


  /**
   * \brief Extract the diagonal from a square matrix.
   * \tparam Arg A diagonal matrix
   * \returns Arg A column vector
   */
#ifdef __cpp_concepts
  template<typename Arg> requires (any_dynamic_dimension<Arg> or square_matrix<Arg>)
#else
  template<typename Arg, std::enable_if_t<any_dynamic_dimension<Arg> or square_matrix<Arg>, int> = 0>
#endif
  inline decltype(auto)
  diagonal_of(Arg&& arg)
  {
    using Scalar = scalar_type_of_t<Arg>;

    constexpr std::size_t dim = dynamic_rows<Arg> ? column_dimension_of_v<Arg> : row_dimension_of_v<Arg>;

    if constexpr (identity_matrix<Arg>)
    {
      return make_constant_matrix_like<1, dim, 1>(arg);
    }
    else if constexpr (zero_matrix<Arg>)
    {
      if constexpr (not square_matrix<Arg>) detail::check_if_square_at_runtime(arg);
      return make_zero_matrix_like<dim, 1>(arg);
    }
    else if constexpr (constant_matrix<Arg> or constant_diagonal_matrix<Arg>)
    {
      if constexpr (not constant_diagonal_matrix<Arg> and not square_matrix<Arg>)
        detail::check_if_square_at_runtime(arg);

      constexpr auto c = []{
        if constexpr (constant_matrix<Arg>) return constant_coefficient_v<Arg>;
        else return constant_diagonal_coefficient_v<Arg>;
      }();

#  if __cpp_nontype_template_args >= 201911L
      return make_constant_matrix_like<c, dim, 1>(arg);
#  else
      constexpr auto c_integral = []{
        if constexpr (std::is_integral_v<decltype(c)>) return c;
        else return static_cast<std::intmax_t>(c);
      }();

      if constexpr (are_within_tolerance(c, static_cast<Scalar>(c_integral)))
        return make_constant_matrix_like<c_integral, dim, 1>(arg);
      else
        return make_self_contained(c * make_constant_matrix_like<1, dim, 1>(arg));
#  endif
    }
    else
    {
      return interface::Conversions<std::decay_t<Arg>>::diagonal_of(std::forward<Arg>(arg));
    }
  }


  // ========================== //
  //  Linear algebra functions  //
  // ========================== //

  /**
   * \brief Take the conjugate of a matrix
   * \tparam Arg The matrix
   */
  template<typename Arg>
  constexpr decltype(auto) conjugate(Arg&& arg) noexcept
  {
    if constexpr (not complex_number<scalar_type_of_t<Arg>> or zero_matrix<Arg> or identity_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      if constexpr (std::imag(constant_coefficient_v<Arg>) == 0)
        return std::forward<Arg>(arg);
      else
        return interface::LinearAlgebra<std::decay_t<Arg>>::conjugate(std::forward<Arg>(arg));
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      if constexpr (std::imag(constant_diagonal_coefficient_v<Arg>) == 0)
        return std::forward<Arg>(arg);
      else
        return interface::LinearAlgebra<std::decay_t<Arg>>::conjugate(std::forward<Arg>(arg));
    }
    else
    {
      return interface::LinearAlgebra<std::decay_t<Arg>>::conjugate(std::forward<Arg>(arg));
    }
  }


  /**
   * \brief Take the transpose of a matrix
   * \tparam Arg The matrix
   */
  template<typename Arg>
  constexpr decltype(auto) transpose(Arg&& arg) noexcept
  {
    if constexpr (diagonal_matrix<Arg> or (self_adjoint_matrix<Arg> and not complex_number<scalar_type_of_t<Arg>>) or
      (constant_matrix<Arg> and square_matrix<Arg>))
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return interface::LinearAlgebra<std::decay_t<Arg>>::transpose(std::forward<Arg>(arg));
    }
  }


  /**
   * \brief Take the adjoint of a matrix
   * \tparam Arg The matrix
   */
  template<typename Arg>
  constexpr decltype(auto) adjoint(Arg&& arg) noexcept
  {
    if constexpr (self_adjoint_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (zero_matrix<Arg> or not complex_number<scalar_type_of_t<Arg>>)
    {
      return transpose(std::forward<Arg>(arg));
    }
    else if constexpr (constant_matrix<Arg>)
    {
      if constexpr (std::imag(constant_coefficient_v<Arg>) == 0)
        return transpose(std::forward<Arg>(arg));
      else if constexpr (not any_dynamic_dimension<Arg> and row_dimension_of_v<Arg> == column_dimension_of_v<Arg>)
        return conjugate(std::forward<Arg>(arg));
      else
        return interface::LinearAlgebra<std::decay_t<Arg>>::adjoint(std::forward<Arg>(arg));
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      if constexpr (std::imag(constant_diagonal_coefficient_v<Arg>) == 0)
        return transpose(std::forward<Arg>(arg));
      else if constexpr (not any_dynamic_dimension<Arg> and row_dimension_of_v<Arg> == column_dimension_of_v<Arg>)
        return conjugate(std::forward<Arg>(arg));
      else
        return interface::LinearAlgebra<std::decay_t<Arg>>::adjoint(std::forward<Arg>(arg));
    }
    else
    {
      return interface::LinearAlgebra<std::decay_t<Arg>>::adjoint(std::forward<Arg>(arg));
    }
  }


  /**
   * \brief Take the determinant of a matrix
   * \tparam Arg The matrix
   */
#ifdef __cpp_concepts
  template<typename Arg> requires any_dynamic_dimension<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<any_dynamic_dimension<Arg> or square_matrix<Arg>, int> = 0>
#endif
  constexpr auto determinant(Arg&& arg)
  {
    if constexpr (any_dynamic_dimension<Arg>) if (runtime_dimension_of<0>(arg) != runtime_dimension_of<1>(arg))
      throw std::domain_error {
        "In determinant, rows of arg (" + std::to_string(runtime_dimension_of<0>(arg)) + ") do not match columns of arg (" +
        std::to_string(runtime_dimension_of<1>(arg)) + ")"};

    using Scalar = scalar_type_of_t<Arg>;

    if constexpr (identity_matrix<Arg>)
    {
      return Scalar(1);
    }
    else if constexpr (zero_matrix<Arg> or (constant_matrix<Arg> and not one_by_one_matrix<Arg>))
    {
      return Scalar(0);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      return constant_coefficient_v<Arg>; //< One-by-one case. General case is handled above.
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      if constexpr (dynamic_rows<Arg>)
        return std::pow(constant_diagonal_coefficient_v<Arg>, runtime_dimension_of<0>(arg));
      else
        return OpenKalman::internal::constexpr_pow(constant_diagonal_coefficient_v<Arg>, row_dimension_of_v<Arg>);
    }
    else if constexpr (one_by_one_matrix<Arg> and element_gettable<Arg, std::size_t, std::size_t>)
    {
      return get_element(arg, std::size_t(0), std::size_t(0));
    }
    else
    {
      auto r = interface::LinearAlgebra<std::decay_t<Arg>>::determinant(std::forward<Arg>(arg));
      static_assert(std::is_convertible_v<decltype(r), const scalar_type_of_t<Arg>>);
      return r;
    }
  }


#ifdef __cpp_concepts
  /**
   * \brief Take the trace of a matrix
   * \tparam Arg The matrix
   */
  template<typename Arg> requires any_dynamic_dimension<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<(any_dynamic_dimension<Arg> or square_matrix<Arg>), int> = 0>
#endif
  constexpr auto trace(Arg&& arg)
  {
    if constexpr (any_dynamic_dimension<Arg>) if (runtime_dimension_of<0>(arg) != runtime_dimension_of<1>(arg))
      throw std::domain_error {
        "In trace, rows of arg (" + std::to_string(runtime_dimension_of<0>(arg)) + ") do not match columns of arg (" +
        std::to_string(runtime_dimension_of<1>(arg)) + ")"};

    using Scalar = scalar_type_of_t<Arg>;

    if constexpr (identity_matrix<Arg>)
    {
      return Scalar(runtime_dimension_of<0>(arg));
    }
    else if constexpr (zero_matrix<Arg>)
    {
      return Scalar(0);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      return Scalar(constant_coefficient_v<Arg> * runtime_dimension_of<0>(arg));
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      return Scalar(constant_diagonal_coefficient_v<Arg> * runtime_dimension_of<0>(arg));
    }
    else if constexpr (one_by_one_matrix<Arg> and element_gettable<Arg, std::size_t, std::size_t>)
    {
      return get_element(arg, std::size_t(0), std::size_t(0));
    }
    else
    {
      auto r = interface::LinearAlgebra<std::decay_t<Arg>>::trace(std::forward<Arg>(arg));
      static_assert(std::is_convertible_v<decltype(r), const scalar_type_of_t<Arg>>);
      return r;
    }
  }


  /**
   * \brief Do a rank update on a matrix, treating it as a self-adjoint matrix.
   * \details If A is not hermitian, the result will modify only the specified storage triangle. The contents of the
   * other elements outside the specified storage triangle are undefined.
   * - The update is A += αUU<sup>*</sup>, returning the updated hermitian A.
   * - If A is an lvalue reference and is writable, it will be updated in place and the return value will be an
   * lvalue reference to the same, updated A. Otherwise, the function returns a new matrix.
   * \tparam t Whether to use the upper triangle elements (TriangleType::upper), lower triangle elements
   * (TriangleType::lower) or diagonal elements (TriangleType::diagonal).
   * \tparam A The matrix to be rank updated.
   * \tparam U The update vector or matrix.
   * \returns an updated native, writable matrix in hermitian form.
   */
#ifdef __cpp_concepts
  template<TriangleType t, typename A, typename U, std::convertible_to<const scalar_type_of_t<A>> Alpha> requires
    (dynamic_rows<U> or dynamic_rows<A> or row_dimension_of_v<U> == row_dimension_of_v<A>) and
    (any_dynamic_dimension<A> or square_matrix<A>)
#else
  template<TriangleType t, typename A, typename U, typename Alpha, std::enable_if_t<
    std::is_convertible_v<Alpha, const scalar_type_of_t<A>> and
    (dynamic_rows<U> or dynamic_rows<A> or
      row_dimension_of<std::decay_t<U>>::value == row_dimension_of<std::decay_t<A>>::value) and
    (any_dynamic_dimension<A> or square_matrix<A>), int> = 0>
#endif
  inline decltype(auto)
  rank_update_self_adjoint(A&& a, U&& u, Alpha alpha = 1)
  {
    if constexpr (dynamic_rows<A> or dynamic_rows<U>) if (runtime_dimension_of<0>(a) != runtime_dimension_of<0>(u))
      throw std::domain_error {
        "In rank_update_self_adjoint, rows of a (" + std::to_string(runtime_dimension_of<0>(a)) + ") do not match rows of u (" +
        std::to_string(runtime_dimension_of<0>(u)) + ")"};

    if constexpr (any_dynamic_dimension<A>) if (runtime_dimension_of<0>(a) != runtime_dimension_of<1>(a))
      throw std::domain_error {
        "In rank_update_self_adjoint, rows of a (" + std::to_string(runtime_dimension_of<0>(a)) + ") do not match columns of a (" +
        std::to_string(runtime_dimension_of<1>(a)) + ")"};

    if constexpr (zero_matrix<U>)
    {
      return std::forward<A>(a);
    }
    // \todo Generalize zero and diagonal cases from eigen and special-matrix
    else
    {
      using Trait = interface::LinearAlgebra<std::decay_t<A>>;
      return Trait::template rank_update_self_adjoint<t>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
  }


  /**
   * \overload
   * \brief The triangle type is derived from A, or is TriangleType::lower by default.
   */
#ifdef __cpp_concepts
  template<typename A, typename U, std::convertible_to<const scalar_type_of_t<A>> Alpha> requires
    (dynamic_rows<U> or dynamic_rows<A> or row_dimension_of_v<U> == row_dimension_of_v<A>) and
    (any_dynamic_dimension<A> or square_matrix<A>)
#else
  template<typename A, typename U, typename Alpha, std::enable_if_t<
    std::is_convertible_v<Alpha, const scalar_type_of_t<A>> and
    (dynamic_rows<U> or dynamic_rows<A> or row_dimension_of<U>::value == row_dimension_of<A>::value) and
    (any_dynamic_dimension<A> or square_matrix<A>), int> = 0>
#endif
  inline decltype(auto)
  rank_update_self_adjoint(A&& a, U&& u, Alpha alpha = 1)
  {
    if constexpr (self_adjoint_matrix<A>)
      return rank_update_self_adjoint<self_adjoint_triangle_type_of_v<A>>(std::forward<A>(a), std::forward<U>(u), alpha);
    else if constexpr (triangular_matrix<A>)
      return rank_update_self_adjoint<triangle_type_of_v<A>>(std::forward<A>(a), std::forward<U>(u), alpha);
    else
      return rank_update_self_adjoint<TriangleType::lower>(std::forward<A>(a), std::forward<U>(u), alpha);
  }


  /**
   * \brief Do a rank update on a matrix, treating it as a triangular matrix.
   * \details If A is not a triangular matrix, the result will modify only the specified triangle. The contents of
   * other elements outside the specified triangle are undefined.
   * - If A is lower-triangular, diagonal, or one-by-one, the update is AA<sup>*</sup> += αUU<sup>*</sup>,
   * returning the updated A.
   * - If A is upper-triangular, the update is A<sup>*</sup>A += αUU<sup>*</sup>, returning the updated A.
   * - If A is an lvalue reference and is writable, it will be updated in place and the return value will be an
   * lvalue reference to the same, updated A. Otherwise, the function returns a new matrix.
   * \tparam t Whether to use the upper triangle elements (TriangleType::upper), lower triangle elements
   * (TriangleType::lower) or diagonal elements (TriangleType::diagonal).
   * \tparam A The matrix to be rank updated.
   * \tparam U The update vector or matrix.
   * \returns an updated native, writable matrix in triangular (or diagonal) form.
   */
# ifdef __cpp_concepts
  template<TriangleType t, typename A, typename U, std::convertible_to<const scalar_type_of_t<A>> Alpha> requires
    (t != TriangleType::lower or lower_triangular_matrix<A> or not upper_triangular_matrix<A>) and
    (t != TriangleType::upper or upper_triangular_matrix<A> or not lower_triangular_matrix<A>) and
    (dynamic_rows<A> or dynamic_rows<U> or row_dimension_of_v<A> == row_dimension_of_v<U>) and
    (any_dynamic_dimension<A> or square_matrix<A>)
# else
  template<TriangleType t, typename A, typename U, typename Alpha, std::enable_if_t<
    std::is_convertible_v<Alpha, const scalar_type_of_t<A>> and
    (t != TriangleType::lower or lower_triangular_matrix<A> or not upper_triangular_matrix<A>) and
    (t != TriangleType::upper or upper_triangular_matrix<A> or not lower_triangular_matrix<A>) and
    (dynamic_rows<A> or dynamic_rows<U> or row_dimension_of<A>::value == row_dimension_of<U>::value) and
    (any_dynamic_dimension<A> or square_matrix<A>), int> = 0>
# endif
  inline decltype(auto)
  rank_update_triangular(A&& a, U&& u, Alpha alpha = 1)
  {
    if constexpr (dynamic_rows<A> or dynamic_rows<U>) if (runtime_dimension_of<0>(a) != runtime_dimension_of<0>(u))
      throw std::domain_error {
        "In rank_update_triangular, rows of a (" + std::to_string(runtime_dimension_of<0>(a)) + ") do not match rows of u (" +
        std::to_string(runtime_dimension_of<0>(u)) + ")"};

    if constexpr (any_dynamic_dimension<A>) if (runtime_dimension_of<0>(a) != runtime_dimension_of<1>(a))
      throw std::domain_error {
        "In rank_update_triangular, rows of a (" + std::to_string(runtime_dimension_of<0>(a)) + ") do not match columns of a (" +
        std::to_string(runtime_dimension_of<1>(a)) + ")"};

    if constexpr (zero_matrix<U>)
    {
      return std::forward<A>(a);
    }
    // \todo Generalize zero and diagonal cases from eigen and special-matrix
    else
    {
      using Trait = interface::LinearAlgebra<std::decay_t<A>>;
      return Trait::template rank_update_triangular<t>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
  }


  /**
   * \overload
   * \brief The triangle type is derived from A, or is TriangleType::lower by default.
   */
# ifdef __cpp_concepts
  template<typename A, typename U, std::convertible_to<const scalar_type_of_t<A>> Alpha> requires
    (dynamic_rows<A> or dynamic_rows<U> or row_dimension_of_v<A> == row_dimension_of_v<U>) and
    (any_dynamic_dimension<A> or square_matrix<A>)
# else
  template<typename A, typename U, typename Alpha, std::enable_if_t<
    std::is_convertible_v<Alpha, const scalar_type_of_t<A>> and
    (dynamic_rows<A> or dynamic_rows<U> or row_dimension_of<A>::value == row_dimension_of<U>::value) and
    (any_dynamic_dimension<A> or square_matrix<A>), int> = 0>
# endif
  inline decltype(auto)
  rank_update_triangular(A&& a, U&& u, Alpha alpha = 1)
  {
    if constexpr (triangular_matrix<A>)
      return rank_update_triangular<triangle_type_of_v<A>>(std::forward<A>(a), std::forward<U>(u), alpha);
    else if constexpr (self_adjoint_matrix<A>)
      return rank_update_triangular<self_adjoint_triangle_type_of_v<A>>(std::forward<A>(a), std::forward<U>(u), alpha);
    else
      return rank_update_triangular<TriangleType::lower>(std::forward<A>(a), std::forward<U>(u), alpha);
  }


  /**
   * \brief Do a rank update on a hermitian, triangular, or one-by-one matrix.
   * \details
   * - If A is hermitian and non-diagonal, then the update is A += αUU<sup>*</sup>, returning the updated hermitian A.
   * - If A is lower-triangular, diagonal, or one-by-one, the update is AA<sup>*</sup> += αUU<sup>*</sup>,
   * returning the updated A.
   * - If A is upper-triangular, the update is A<sup>*</sup>A += αUU<sup>*</sup>, returning the updated A.
   * - If A is an lvalue reference and is writable, it will be updated in place and the return value will be an
   * lvalue reference to the same, updated A. Otherwise, the function returns a new matrix.
   * \tparam A A matrix to be updated, which must be hermitian (known at compile time),
   * triangular (known at compile time), or one-by-one (determinable at runtime) .
   * \tparam U A matrix or column vector with a number of rows that matches the size of A.
   * \param alpha A scalar multiplication factor.
   */
#ifdef __cpp_concepts
  template<typename A, typename U, std::convertible_to<const scalar_type_of_t<A>> Alpha> requires
    (triangular_matrix<A> or self_adjoint_matrix<A> or (zero_matrix<A> and any_dynamic_dimension<A>) or
      (constant_matrix<A> and not complex_number<scalar_type_of<A>> and any_dynamic_dimension<A>) or
      (dynamic_rows<A> and dynamic_columns<A>) or (dynamic_rows<A> and column_vector<A>) or
      (dynamic_columns<A> and row_vector<A>)) and
    (dynamic_rows<U> or dynamic_rows<A> or row_dimension_of_v<U> == row_dimension_of_v<A>) and
    (any_dynamic_dimension<A> or square_matrix<A>)
#else
  template<typename A, typename U, typename Alpha, std::enable_if_t<
    std::is_convertible_v<Alpha, const scalar_type_of_t<A>> and
    (triangular_matrix<A> or self_adjoint_matrix<A> or (zero_matrix<A> and any_dynamic_dimension<A>) or
      (constant_matrix<A> and not complex_number<scalar_type_of<A>> and any_dynamic_dimension<A>) or
      (dynamic_rows<A> and dynamic_columns<A>) or (dynamic_rows<A> and column_vector<A>) or
      (dynamic_columns<A> and row_vector<A>)) and
    (dynamic_rows<U> or dynamic_rows<A> or row_dimension_of<U>::value == row_dimension_of<A>::value) and
    (any_dynamic_dimension<A> or square_matrix<A>), int> = 0>
#endif
  inline decltype(auto)
  rank_update(A&& a, U&& u, Alpha alpha = 1)
  {
    if constexpr (dynamic_rows<A> or dynamic_rows<U>) if (runtime_dimension_of<0>(a) != runtime_dimension_of<0>(u))
      throw std::domain_error {
        "In rank_update, rows of a (" + std::to_string(runtime_dimension_of<0>(a)) + ") do not match rows of u (" +
        std::to_string(runtime_dimension_of<0>(u)) + ")"};

    if constexpr (triangular_matrix<A>)
    {
      constexpr TriangleType t = triangle_type_of_v<A>;
      return rank_update_triangular<t>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
    else if constexpr (zero_matrix<A> and any_dynamic_dimension<A>)
    {
      return rank_update_triangular<TriangleType::diagonal>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
    else if constexpr (self_adjoint_matrix<A>)
    {
      constexpr TriangleType t = self_adjoint_triangle_type_of_v<A>;
      return rank_update_self_adjoint<t>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
    else if constexpr (constant_matrix<A> and not complex_number<scalar_type_of<A>> and any_dynamic_dimension<A>)
    {
      return rank_update_self_adjoint<TriangleType::lower>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
    else
    {
      if constexpr (any_dynamic_dimension<A>)
        if ((dynamic_rows<A> and runtime_dimension_of<0>(a) != 1) or (dynamic_columns<A> and runtime_dimension_of<1>(a) != 1))
          throw std::domain_error {
            "Non hermitian, non-triangular argument to rank_update expected to be one-by-one, but instead it has " +
            std::to_string(runtime_dimension_of<0>(a)) + " rows and " + std::to_string(runtime_dimension_of<1>(a)) + " columns"};

      auto e = std::sqrt(trace(a) * trace(a.conjugate()) + alpha * trace(u * adjoint(u)));

      if constexpr (std::is_lvalue_reference_v<A> and not std::is_const_v<std::remove_reference_t<A>> and writable<A>)
      {
        if constexpr (element_settable<A, std::size_t, std::size_t>)
        {
          set_element(a, e, 0, 0);
        }
        else
        {
          a = MatrixTraits<A>::make(e);
        }
        return std::forward<A>(a);
      }
      else
      {
        return MatrixTraits<A>::make(e);
      }
    }
  }


  // ==================== //
  //  Internal functions  //
  // ==================== //

  namespace internal
  {
    // ------------------------ //
    //  to_covariance_nestable  //
    // ------------------------ //

    /**
     * \overload
     * \internal
     * \brief Convert a \ref covariance_nestable matrix or \ref typed_matrix_nestable to a \ref covariance_nestable.
     * \tparam T \ref covariance_nestable to which Arg is to be converted.
     * \return A \ref covariance_nestable of type T.
     */
#ifdef __cpp_concepts
    template<covariance_nestable T, typename Arg> requires
      (covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_matrix<Arg> or column_vector<Arg>))) and
      (row_dimension_of_v<Arg> == row_dimension_of_v<T>) and
      (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
      (not diagonal_matrix<T> or diagonal_matrix<Arg> or column_vector<Arg>)
#else
    template<typename T, typename Arg, typename = std::enable_if_t<
      (not std::is_same_v<T, Arg>) and covariance_nestable<T> and
      (covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_matrix<Arg> or column_vector<Arg>))) and
      (row_dimension_of<Arg>::value == row_dimension_of<T>::value) and
      (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
      (not diagonal_matrix<T> or diagonal_matrix<Arg> or column_vector<Arg>)>>
#endif
    constexpr decltype(auto)
    to_covariance_nestable(Arg&&) noexcept;


    /**
     * \internal
     * \brief Convert \ref covariance or \ref typed_matrix to a \ref covariance_nestable of type T.
     * \tparam T \ref covariance_nestable to which Arg is to be converted.
     * \tparam Arg A \ref covariance or \ref typed_matrix.
     * \return A \ref covariance_nestable of type T.
     */
#ifdef __cpp_concepts
    template<covariance_nestable T, typename Arg> requires
      (covariance<Arg> or (typed_matrix<Arg> and (square_matrix<Arg> or column_vector<Arg>))) and
      (row_dimension_of_v<Arg> == row_dimension_of_v<T>) and
      (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
      (not diagonal_matrix<T> or diagonal_matrix<Arg> or column_vector<Arg>)
#else
    template<typename T, typename Arg, typename = void, typename = std::enable_if_t<
      (not std::is_same_v<T, Arg>) and covariance_nestable<T> and (not std::is_void_v<Arg>) and
      (covariance<Arg> or (typed_matrix<Arg> and (square_matrix<Arg> or column_vector<Arg>))) and
      (row_dimension_of<Arg>::value == row_dimension_of<T>::value) and
      (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
      (not diagonal_matrix<T> or diagonal_matrix<Arg> or column_vector<Arg>)>>
#endif
    constexpr decltype(auto)
    to_covariance_nestable(Arg&&) noexcept;


    /**
     * \overload
     * \internal
     * /return The result of converting Arg to a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<typename Arg>
    requires covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_matrix<Arg> or column_vector<Arg>))
#else
    template<typename Arg, typename = std::enable_if_t<covariance_nestable<Arg> or
        (typed_matrix_nestable<Arg> and (square_matrix<Arg> or column_vector<Arg>))>>
#endif
    constexpr decltype(auto)
    to_covariance_nestable(Arg&&) noexcept;


    /**
     * \overload
     * \internal
     * /return A \ref triangular_matrix if Arg is a \ref triangular_covariance or otherwise a \ref self_adjoint_matrix.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires covariance<Arg> or
      (typed_matrix<Arg> and (square_matrix<Arg> or column_vector<Arg>))
#else
    template<typename Arg, typename = void, typename = std::enable_if_t<covariance<Arg> or
      (typed_matrix<Arg> and (square_matrix<Arg> or column_vector<Arg>))>>
#endif
    constexpr decltype(auto)
    to_covariance_nestable(Arg&&) noexcept;

  } // namespace internal


} // namespace OpenKalman

#endif //OPENKALMAN_FUNCTIONS_HPP
