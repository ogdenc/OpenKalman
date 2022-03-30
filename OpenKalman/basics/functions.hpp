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

  // ------------------- //
  //  get_dimensions_of  //
  // ------------------- //

#ifdef __cpp_concepts
  template<std::size_t N, indexible Arg> requires (N < max_indices_of_v<Arg>)
#else
  template<std::size_t N, typename Arg, std::enable_if_t<indexible<Arg> and N < max_indices_of<T>::value, int> = 0>
#endif
  constexpr auto get_dimensions_of(const Arg& arg)
  {
    if constexpr (dynamic_dimension<Arg, N>)
      return Dimensions{interface::IndexTraits<std::decay_t<Arg>, N>::dimension_at_runtime(arg)};
    else
      return Dimensions<interface::IndexTraits<std::decay_t<Arg>, N>::dimension>{};
  }


  // ---------------------- //
  //  runtime_dimension_of  //
  // ---------------------- //

#ifdef __cpp_concepts
  template<std::size_t N = 0, indexible Arg> requires (N < max_indices_of_v<Arg>)
#else
  template<std::size_t N = 0, typename Arg, std::enable_if_t<indexible<Arg> and N < max_indices_of<T>::value, int> = 0>
#endif
  constexpr std::size_t runtime_dimension_of(const Arg& arg)
  {
    return interface::IndexTraits<std::decay_t<Arg>, N>::dimension_at_runtime(arg);
  }


  // -------------------------------- //
  //  get_coordinate_system_types_of  //
  // -------------------------------- //

  template<std::size_t N = 0, typename Arg>
  constexpr auto get_coordinate_system_types_of(Arg&& arg)
  {
    return interface::CoordinateSystemTraits<std::decay_t<Arg>, N>::coordinate_system_types_at_runtime(std::forward<Arg>(arg));
  }


  // --------------------- //
  //  get_tensor_order_of  //
  // --------------------- //

  namespace detail
  {
    template<std::size_t...I, typename T>
    constexpr auto get_tensor_order_of_impl(std::index_sequence<I...>, const T& t)
    {
      return ((runtime_dimension_of<I>(t) == 1 ? 0 : 1) + ... + 0);
    }
  }


  /**
   * \brief Return a tuple of \ref index_descriptor objects defining the dimensions of T.
   * \tparam T A matrix or array
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr auto get_tensor_order_of(const T& t)
  {
    if constexpr (not any_dynamic_dimension<T>)
      return tensor_order_of_v<T>;
    else
      return detail::get_tensor_order_of_impl(std::make_index_sequence<max_indices_of_v<T>>{}, t);
  }


  // ----------------------- //
  //  get_all_dimensions_of  //
  // ----------------------- //

  namespace detail
  {
    template<typename T, std::size_t...I>
    constexpr auto get_all_dimensions_of_impl(std::index_sequence<I...>, const T& t)
    {
      return std::tuple {[](const T& t){
        constexpr std::size_t size = index_dimension_of_v<T, I>;
        if constexpr (size == dynamic_size)
          return Dimensions<size>{runtime_dimension_of<I>(t)};
        else
          return Dimensions<size>{};
      }(t)...};
    }


    template<typename T, std::size_t...I>
    constexpr auto get_all_dimensions_of_impl(std::index_sequence<I...>)
    {
      return std::tuple {Dimensions<index_dimension_of_v<T, I>>{}...};
    }
  }


  /**
   * \brief Return a tuple of \ref index_descriptor objects defining the dimensions of T.
   * \tparam T A matrix or array
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr auto get_all_dimensions_of(const T& t)
  {
    return detail::get_all_dimensions_of_impl(std::make_index_sequence<max_indices_of_v<T>>{}, t);
  }


  /**
   * \overload
   * \brief Return a tuple of \ref index_descriptor objects defining the dimensions of T.
   * \details This overload is only enabled if all dimensions of T are known at compile time.
   * \tparam T A matrix or array
   */
#ifdef __cpp_concepts
  template<indexible T> requires (not any_dynamic_dimension<T>)
#else
  template<typename T, std::enable_if_t<indexible<T> and not any_dynamic_dimension<T>, int> = 0>
#endif
  constexpr auto get_all_dimensions_of()
  {
    return detail::get_all_dimensions_of_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{});
  }


  namespace internal
  {

    // ----------------------- //
    //  make_dimensions_tuple  //
    // ----------------------- //

    namespace detail
    {
      template<typename T, std::size_t...Is>
      constexpr std::size_t count_dynamic_dimensions(std::index_sequence<Is...>)
      {
        return ((index_dimension_of_v<T, Is> == dynamic_size ? 1 : 0) + ... + 0);
      }


      template<typename T, std::size_t I_begin, typename D, std::size_t...Is>
      constexpr auto iterate_dimensions_tuple(D&& d, std::index_sequence<Is...>)
      {
        return std::tuple {std::forward<D>(d), Dimensions<index_dimension_of_v<T, I_begin + Is>>{}...};
      }


      template<typename T, std::size_t I, std::size_t Max, typename N, typename...Ns>
      constexpr auto make_dimensions_tuple_impl(N&& n, Ns&&...ns)
      {
        constexpr auto dim_I = index_dimension_of_v<T, I>;
        constexpr std::size_t next_I = I + 1;

        if constexpr (next_I >= Max)
        {
          if constexpr (dim_I == dynamic_size)
            return std::tuple {Dimensions{std::forward<N>(n)}};
          else
            return std::tuple {Dimensions<dim_I>{}};
        }
        else if constexpr (sizeof...(Ns) == 0)
        {
          if constexpr (dim_I == dynamic_size)
            return iterate_dimensions_tuple<T, next_I>(
              Dimensions{std::forward<N>(n)}, std::make_index_sequence<Max - next_I>{});
          else
            return iterate_dimensions_tuple<T, next_I>(Dimensions<dim_I>{}, std::make_index_sequence<Max - next_I>{});
        }
        else if constexpr (dim_I == dynamic_size)
        {
          return std::tuple_cat(std::tuple {Dimensions{std::forward<N>(n)}},
            make_dimensions_tuple_impl<T, next_I, Max>(std::forward<Ns>(ns)...));
        }
        else
        {
          return std::tuple_cat(std::tuple {Dimensions<dim_I>{}},
            make_dimensions_tuple_impl<T, next_I, Max>(std::forward<N>(n), std::forward<Ns>(ns)...));
        }
      }
    }


#ifdef __cpp_concepts
    template<indexible T, std::convertible_to<std::size_t> ... N> requires
     (sizeof...(N) == detail::count_dynamic_dimensions<T>(std::make_index_sequence<max_indices_of_v<T>> {}))
#else
    template<typename T, typename...N, std::enable_if_t<indexible<T> and
      (std::is_convertible_v<N, std::size_t> and ...) and
      (sizeof...(N) == detail::count_dynamic_dimensions<T>(std::make_index_sequence<max_indices_of_v<T>> {})), int> = 0>
#endif
    constexpr auto make_dimensions_tuple(N&&...n)
    {
      return detail::make_dimensions_tuple_impl<T, 0, max_indices_of_v<T>>(std::forward<N>(n)...);
    }

  } // namespace internal


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
    constexpr auto rows = dynamic_rows<Arg> and (self_adjoint_matrix<Arg> or triangular_matrix<Arg>) ?
      column_dimension_of_v<Arg> : row_dimension_of_v<Arg>;
    constexpr auto cols = dynamic_columns<Arg> and (self_adjoint_matrix<Arg> or triangular_matrix<Arg>) ?
      row_dimension_of_v<Arg> : column_dimension_of_v<Arg>;

    using Trait = EquivalentDenseWritableMatrix<std::decay_t<Arg>, rows, cols>;
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
   * \brief Make a default, dense, writable matrix based on a list of Dimensions describing the sizes of each index.
   * \tparam T A dummy matrix or array from the relevant library (size and shape does not matter)
   * \param d a tuple of Dimensions describing the sizes of each index
   */
#ifdef __cpp_concepts
  template<indexible T, typename Scalar = scalar_type_of_t<T>, index_descriptor...D> requires
    (sizeof...(D) == max_indices_of_v<T>)
#else
  template<typename T, typename Scalar = scalar_type_of_t<T>, typename...D,
    std::enable_if_t<indexible<T> and (index_descriptor<D> and ...) and sizeof...(D) == max_indices_of_v<T>, int> = 0>
#endif
  constexpr decltype(auto)
  make_default_dense_writable_matrix_like(D&&...d)
  {
    return EquivalentDenseWritableMatrix<T, dimension_size_of_v<D>..., Scalar>::make_default(std::forward<D>(d)...);
  }


  /**
   * \overload
   * \brief Make a default, dense, writable matrix based on a list of Dimensions describing the sizes of each index.
   * \tparam T A pattern matrix or array having dimensions known fully at compile time
   */
#ifdef __cpp_concepts
  template<indexible T, typename Scalar = scalar_type_of_t<T>> requires (not any_dynamic_dimension<T>)
#else
  template<typename T, typename Scalar = scalar_type_of_t<T>,
    std::enable_if_t<indexible<T> and not any_dynamic_dimension<T>, int> = 0>
#endif
  constexpr decltype(auto)
  make_default_dense_writable_matrix_like()
  {
    using Trait = EquivalentDenseWritableMatrix<T, index_dimension_of_v<T, 0>, index_dimension_of_v<T, 1>, Scalar>;
    return std::apply([](auto&&...d) { return Trait::make_default(std::forward<decltype(d)>(d)...); },
      get_all_dimensions_of<T>());
  }


  /**
   * \overload
   * \brief Make a default, dense, writable matrix based on an existing object.
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
    using Trait = EquivalentDenseWritableMatrix<T, rows, columns, Scalar>;
    return std::apply([](auto&&...d) { return Trait::make_default(std::forward<decltype(d)>(d)...); },
      get_all_dimensions_of(t));
  }


  /**
   * \overload
   * \brief Make a default, dense, writable matrix based on an existing object, but specifying new dimensions.
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
   * \details The new matrix will be one in which basic matrix operations are defined.
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

#ifdef __cpp_concepts
  template<indexible T, auto constant, typename Scalar = scalar_type_of_t<T>, index_descriptor...D> requires
    (sizeof...(D) == max_indices_of_v<T>)
#else
  template<typename T, typename Scalar = scalar_type_of_t<T>, typename...D,
    std::enable_if_t<indexible<T> and (index_descriptor<D> and ...) and sizeof...(D) == max_indices_of_v<T>, int> = 0>
#endif
  constexpr decltype(auto)
  make_constant_matrix_like(D&&...d)
  {
    return SingleConstantMatrixTraits<T, dimension_size_of_v<D>..., Scalar>::template make_constant_matrix<constant>(
      std::forward<D>(d)...);
  }


  /**
   * \overload
   * \brief Make a single-constant matrix based on T, but specifying a new constant.
   * \tparam T The matrix or array on which the new matrix is patterned.
   * \tparam constant The constant.
   * \tparam Scalar A scalar type for the new matrix.
   */
#ifdef __cpp_concepts
  template<auto constant, typename Scalar, indexible T>
#else
  template<auto constant, typename Scalar, typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr decltype(auto)
  make_constant_matrix_like(T&& t)
  {
    constexpr bool constants_match = []{
      if constexpr (constant_matrix<T>) return are_within_tolerance(constant, constant_coefficient_v<T>);
      else return false;
    }();

    if constexpr (constants_match and std::is_same_v<scalar_type_of_t<T>, Scalar>)
      return std::forward<T>(t);
    else
      return std::apply(
        [](auto&&...arg){ return make_constant_matrix_like<T, constant, Scalar>(std::forward<decltype(arg)>(arg)...); },
        get_all_dimensions_of(t));
  }


  /**
   * \overload
   * \brief Make a single-constant matrix based on T, but specifying a new constant (of scalar type the same as T).
   * \tparam T The matrix or array on which the new matrix is patterned.
   * \tparam constant The constant.
   */
#ifdef __cpp_concepts
  template<auto constant, indexible T>
#else
  template<auto constant, typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr decltype(auto)
  make_constant_matrix_like(T&& t)
  {
    return make_constant_matrix_like<constant, scalar_type_of_t<T>>(std::forward<T>(t));
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
  constexpr decltype(auto) get_element(Arg&& arg, const I...i)
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


  // =================== //
  //  Subset operations  //
  // =================== //

  /**
   * \brief Extract one column from a matrix or other tensor.
   * \details The index of the column may be specified at either compile time <em>or</em> at runtime, but not both.
   * \tparam compile_time_index The index of the column, if specified at compile time
   * \tparam Arg The matrix or other tensor from which the column is to be extracted
   * \tparam runtime_index_t The type of the index of the column, if the index is specified at runtime. This type
   * should be convertible to <code>std::size_t</code>
   * \return A \ref column_vector
   */
#ifdef __cpp_concepts
  template<std::size_t...compile_time_index, typename Arg, std::convertible_to<const std::size_t>...runtime_index_t> requires
    (sizeof...(compile_time_index) + sizeof...(runtime_index_t) == 1) and
    (dynamic_columns<Arg> or ((compile_time_index + ... + 0) < column_dimension_of_v<Arg>))
#else
  template<std::size_t...compile_time_index, typename Arg, typename...runtime_index_t, std::enable_if_t<
    (std::is_convertible_v<runtime_index_t, const std::size_t> and ...) and
    (sizeof...(compile_time_index) + sizeof...(runtime_index_t) == 1) and
    (dynamic_columns<Arg> or ((compile_time_index + ... + 0) < column_dimension_of<Arg>::value)), int> = 0>
#endif
  constexpr decltype(auto)
  column(Arg&& arg, runtime_index_t...i)
  {
    if constexpr (column_vector<Arg>)
    {
      if constexpr (sizeof...(i) == 1)
      {
        if ((i + ... + 0) != 0)
          throw std::out_of_range {"Runtime column index (which is " + std::to_string((i + ... + 0)) +
            ") is not in range 0 <= i < 1."};
        else
          return std::forward<Arg>(arg);
      }
      else if constexpr ((compile_time_index + ... + 0) != 0)
      {
        throw std::out_of_range {"Compile-time column index (which is " +
          std::to_string((compile_time_index + ... + 0)) + ") is not in range 0 <= i < 1."};      }
      else
      {
        return std::forward<Arg>(arg);
      }
    }
    else if constexpr (constant_matrix<Arg>)
    {
      auto runtime_i = (compile_time_index + ... + (i + ... + 0));
      auto cols = runtime_dimension_of<1>(arg);

      if (runtime_i >= cols) throw std::out_of_range {"Runtime column index (which is " + std::to_string(runtime_i) +
          ") is not in range 0 <= i < " + std::to_string(cols) + "."};

      if constexpr (zero_matrix<Arg>)
        return make_zero_matrix_like<row_dimension_of_v<Arg>, 1>(std::forward<Arg>(arg));
      else
        return make_constant_matrix_like<Arg, constant_coefficient_v<Arg>>(get_dimensions_of<0>(arg), Dimensions<1>{});
    }
    else
    {
      return interface::Subsets<std::decay_t<Arg>>::template column<compile_time_index...>(std::forward<Arg>(arg), i...);
    }
  }


  /**
   * \brief Extract one row from a matrix or other tensor.
   * \details The index of the row may be specified at either compile time <em>or</em> at runtime, but not both.
   * \tparam compile_time_index The index of the row, if specified at compile time
   * \tparam Arg The matrix or other tensor from which the row is to be extracted
   * \tparam runtime_index_t The type of the index of the row, if the index is specified at runtime. This type
   * should be convertible to <code>std::size_t</code>
   * \return A \ref row_vector
   */
#ifdef __cpp_concepts
  template<std::size_t...compile_time_index, typename Arg, std::convertible_to<const std::size_t>...runtime_index_t> requires
    (sizeof...(compile_time_index) + sizeof...(runtime_index_t) == 1) and
    (dynamic_rows<Arg> or ((compile_time_index + ... + 0) < row_dimension_of_v<Arg>))
#else
  template<size_t...compile_time_index, typename Arg, typename...runtime_index_t, std::enable_if_t<
    (std::is_convertible_v<runtime_index_t, const std::size_t> and ...) and
    (sizeof...(compile_time_index) + sizeof...(runtime_index_t) == 1) and
    (dynamic_rows<Arg> or ((compile_time_index + ... + 0) < row_dimension_of<Arg>::value)), int> = 0>
#endif
  constexpr decltype(auto)
  row(Arg&& arg, runtime_index_t...i)
  {
    if constexpr (row_vector<Arg>)
    {
      if constexpr (sizeof...(i) == 1)
      {
        if ((i + ... + 0) != 0)
          throw std::out_of_range {"Runtime row index (which is " + std::to_string((i + ... + 0)) +
            ") is not in range 0 <= i < 1."};
        else
          return std::forward<Arg>(arg);
      }
      else if constexpr ((compile_time_index + ... + 0) != 0)
      {
        throw std::out_of_range {"Compile-time row index (which is " +
          std::to_string((compile_time_index + ... + 0)) + ") is not in range 0 <= i < 1."};      }
      else
      {
        return std::forward<Arg>(arg);
      }
    }
    else if constexpr (constant_matrix<Arg>)
    {
      auto runtime_i = (compile_time_index + ... + (i + ... + 0));
      auto rows = runtime_dimension_of<0>(arg);

      if (runtime_i >= rows) throw std::out_of_range {"Runtime row index (which is " + std::to_string(runtime_i) +
          ") is not in range 0 <= i < " + std::to_string(rows) + "."};

      if constexpr (zero_matrix<Arg>)
        return make_zero_matrix_like<1, column_dimension_of_v<Arg>>(std::forward<Arg>(arg));
      else
        return make_constant_matrix_like<Arg, constant_coefficient_v<Arg>>(Dimensions<1>{}, get_dimensions_of<1>(arg));
    }
    else
    {
      return interface::Subsets<std::decay_t<Arg>>::template row<compile_time_index...>(std::forward<Arg>(arg), i...);
    }
  }


  // ================== //
  //  Array operations  //
  // ================== //

  namespace detail
  {
    template<typename T, typename Arg, std::size_t...indices>
    constexpr bool check_n_ary_dims_impl(std::index_sequence<indices...>)
    {
      return ((
        index_dimension_of_v<Arg, indices> == dynamic_size or
        index_dimension_of_v<T, indices> == dynamic_size or
        index_dimension_of_v<Arg, indices> == 1 or
        index_dimension_of_v<Arg, indices> == index_dimension_of_v<T, indices>) and ...);
    }


    template<typename T, typename Arg>
    constexpr bool check_n_ary_dimensions()
    {
      return check_n_ary_dims_impl<T, Arg>(std::make_index_sequence<max_indices_of_v<T>>());
    }


    template<std::size_t...sizes, typename Arg, std::size_t...indices>
    inline void check_n_ary_rt_dims_impl(
      const std::tuple<Dimensions<sizes>...>& d, const Arg& arg, std::index_sequence<indices...>)
    {
      ((runtime_dimension_of<indices>(arg) == 1 or
        runtime_dimension_of<indices>(arg) == std::get<indices>(d)() ? 0 : throw std::logic_error {
          "In an argument to n_ary_operation_with_broadcasting, the dimension of index " +
          std::to_string(indices) + " is " + std::to_string(runtime_dimension_of<indices>(arg)) + ", but should be 1 " +
          (std::get<indices>(d)() == 1 ? "" : "or " + std::to_string(std::get<indices>(d)())) +
          "(the dimension of index " + std::to_string(indices) + " of the PatternMatrix template argument)"}),...);
    }


    template<typename PatternMatrix, std::size_t...sizes, typename Arg>
    inline void check_n_ary_runtime_dimensions(const std::tuple<Dimensions<sizes>...>& d, const Arg& arg)
    {
      check_n_ary_rt_dims_impl(d, arg, std::make_index_sequence<max_indices_of_v<PatternMatrix>>());
    }


    template<std::size_t I, typename Arg, typename...Args>
    constexpr std::size_t find_max_runtime_dims_impl(const Arg& arg, const Args&...args)
    {
      if constexpr (sizeof...(Args) == 0)
      {
        return runtime_dimension_of<I>(arg);
      }
      else
      {
        auto dim0 = runtime_dimension_of<I>(arg);
        auto dim = find_max_runtime_dims_impl<I>(args...);
        if (dim0 == dim or dim == 1)
          return dim0;
        else if (dim0 == 1)
          return dim;
        else
          throw std::logic_error {"In an argument to n_ary_operation_with_broadcasting, the dimension of index " +
            std::to_string(I) + " is " + std::to_string(dim0) + ", which is not 1 and does not match index " +
            std::to_string(I) + " of a later argument, which is " + std::to_string(dim)};
      }
    }


    template<typename...Args, std::size_t...I>
    constexpr std::size_t find_max_dims_impl(const Args&...args, std::index_sequence<I...>)
    {
      return std::tuple {
        [](const Args&...args){
          constexpr auto max_stat_dim = std::max({(dynamic_dimension<Args, I> ? 0 : index_dimension_of_v<Args, I>)...});
          constexpr auto dim = max_stat_dim == 0 ? dynamic_size : max_stat_dim;

          if constexpr (((not dynamic_dimension<Args, I> and (index_dimension_of_v<Args, I> == 0 or
              (index_dimension_of_v<Args, I> != 1 and index_dimension_of_v<Args, I> != dim))) or ...))
            throw std::logic_error {"In an argument to n_ary_operation_with_broadcasting, the dimension of index " +
              std::to_string(I) + " is " + std::to_string(index_dimension_of_v<Args, I>) + ", but should be 1 or " +
              (dim == 1 or dim == dynamic_size ?
                "the maximum index " + std::to_string(I) + " among the arguments" :
                std::to_string(dim) + "(the maximum index " + std::to_string(I) + " among the arguments)")};

          if constexpr ((dim != dynamic_size and dim > 1) or (dim == 1 and not (dynamic_dimension<Args, I> or ...)))
            return Dimensions<dim>{};
          else
            return Dimensions<dynamic_size>{find_max_runtime_dims_impl<I>(args...)};
        }(args...)...};
    }


    template<typename...Args>
    inline std::size_t find_max_dims(const Args&...args)
    {
      constexpr auto max_indices = std::max({max_indices_of_v<Args>...});
      return find_max_dims_impl(args..., std::make_index_sequence<max_indices>());
    }
  }


  /**
   * \brief Perform a component-wise n-ary operation, using broadcasting to match the size of a pattern matrix.
   * \tparam PatternMatrix A matrix or array of the size and shape corresponding to the result
   * \tparam Operation The n-ary operation taking n arguments
   * \tparam Args Any arguments other than the first
   * \return A matrix or array in which each component is the result of calling Operation on corresponding components
   * from each of the arguments, in the order specified.
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, std::size_t...sizes, typename Operation, indexible...Args> requires
    std::is_invocable_r_v<scalar_type_of_t<PatternMatrix>, Operation&&, typename scalar_type_of<Args>::type...> and
    ((max_indices_of_v<Args> >= max_indices_of_v<PatternMatrix>) and ...) and
    (detail::check_n_ary_dimensions<PatternMatrix, Args>() and ...)
#else
  template<typename PatternMatrix, std::size_t...sizes, typename Operation, typename...Args, std::enable_if_t<
    indexible<PatternMatrix> and (indexible<Args> and ...) and
    std::is_invocable_r<scalar_type_of_t<PatternMatrix>, Operation&&, typename scalar_type_of<Args>::type...>::value and
    ((max_indices_of_v<Args> >= max_indices_of_v<PatternMatrix>) and ...) and
    (detail::check_n_ary_dimensions<PatternMatrix, Args>() and ...), int> = 0>
#endif
  constexpr decltype(auto)
  n_ary_operation_with_broadcasting(const std::tuple<Dimensions<sizes>...>& d, Operation&& op, Args&&...args)
  {
    if constexpr ((any_dynamic_dimension<PatternMatrix> or ... or any_dynamic_dimension<Args>))
      (detail::check_n_ary_runtime_dimensions<PatternMatrix>(d, args),...);

    return interface::ArrayOperations<PatternMatrix>::n_ary_operation_with_broadcasting(
      d, std::forward<Operation>(op), std::forward<Args>(args)...);
  }


  /**
   * \overload
   * \brief Perform a component-wise n-ary operation, using broadcasting if necessary to make the arguments the same size.
   * \details Each of the arguments may be expanded by broadcasting. The result will derive its dimensions from the
   * largest dimensions among the arguments.
   * \tparam Operation The n-ary operation taking n arguments
   * \tparam Arg The first of n arguments
   * \tparam Args Any arguments other than the first
   * \return A matrix or array in which each component is the result of calling Operation on corresponding components
   * from each of the arguments, in the order specified.
   */
#ifdef __cpp_concepts
  template<typename Operation, indexible Arg, indexible...Args> requires
    std::is_invocable_r_v<scalar_type_of_t<Arg>, Operation&&, typename scalar_type_of<Arg>::type,
      typename scalar_type_of<Args>::type...> and
    ((max_indices_of_v<Args> >= max_indices_of_v<Arg>) and ...) and (detail::check_n_ary_dimensions<Arg, Args>() and ...)
#else
  template<typename Operation, typename Arg, typename...Args, std::enable_if_t<
    (indexible<Arg> and ... and indexible<Args>) and
    std::is_invocable_r<scalar_type_of_t<Arg>, Operation&&, typename scalar_type_of<Arg>::type,
      typename scalar_type_of<Args>::type...>::value and
    ((max_indices_of_v<Args> >= max_indices_of_v<Arg>) and ...) and
    (detail::check_n_ary_dimensions<Arg, Args>() and ...), int> = 0>
#endif
  constexpr decltype(auto)
  n_ary_operation_with_broadcasting(Operation&& op, Arg&& arg, Args&&...args)
  {
    auto d = detail::find_max_dims(arg, args...);
    using PatternMatrix = decltype(make_default_dense_writable_matrix_like<Arg>(d));

    return interface::ArrayOperations<PatternMatrix>::n_ary_operation_with_broadcasting(
      d, std::forward<Operation>(op), std::forward<Arg>(arg), std::forward<Args>(args)...);
  }


  namespace detail
  {

    template<auto>
    struct is_constexpr_value : std::true_type {};


#ifdef __cpp_concepts
    template<typename Scalar, decltype(auto) f>
#else
    template<typename Scalar, decltype(auto) f, typename = void>
#endif
    struct is_constexpr_binary_function : std::false_type {};


#ifdef __cpp_concepts
    template<typename Scalar, decltype(auto) f> requires
      is_constexpr_value<static_cast<const bool>(f(static_cast<Scalar>(0), static_cast<Scalar>(0)))>::value
    struct is_constexpr_binary_function<Scalar, f>
#else
    template<typename Scalar, decltype(auto) f>
    struct is_constexpr_binary_function<Scalar, f, std::enable_if_t<
      is_constexpr_value<static_cast<const bool>f(static_cast<Scalar>(0), static_cast<Scalar>(0)))>::value>>
#endif
      : std::true_type {};


    template<typename T>
    struct is_plus : std::false_type {};

    template<typename T>
    struct is_plus<std::plus<T>> : std::true_type {};

    template<typename T>
    struct is_multiplies : std::false_type {};

    template<typename T>
    struct is_multiplies<std::multiplies<T>> : std::true_type {};


    template<typename BinaryFunction, typename C>
    constexpr auto calc_const_reduce_constant_runtime(const std::size_t dim, const BinaryFunction& b, const C& c)
    {
      if (dim <= 1)
        return c;
      else if constexpr (is_plus<BinaryFunction>::value)
        return c * dim;
      else if constexpr (is_multiplies<BinaryFunction>::value)
        return std::pow(c, dim);
      else
        return b(c, calc_const_reduce_constant_runtime(dim - 1, b, c));
    }


    template<std::size_t dim, typename BinaryFunction, typename C>
    constexpr auto calc_const_reduce_constant(const BinaryFunction& b, const C c)
    {
      if constexpr (dim <= 1)
        return c;
      else if constexpr (is_plus<BinaryFunction>::value)
        return c * dim;
      else if constexpr (is_multiplies<BinaryFunction>::value)
        return OpenKalman::internal::constexpr_pow(c, dim);
      else
        return b(c, calc_const_reduce_constant<dim - 1>(b, c));
    }


    template<typename Scalar, std::size_t index, typename BinaryFunction, typename T, std::size_t...I>
    constexpr auto const_reduce(const BinaryFunction& b, T&& t, std::index_sequence<I...>)
    {
      constexpr auto dim = index_dimension_of_v<T, index>;

      if constexpr (dim != dynamic_size and detail::is_constexpr_binary_function<Scalar, b>::value)
      {
        constexpr auto c = calc_const_reduce_constant<dim>(b, constant_coefficient_v<T>);

# if __cpp_nontype_template_args >= 201911L
        return make_constant_matrix_like<T, c>(
          [](const T& t){
            if constexpr (index == I)
              return Dimensions<1>{};
            else
              return std::get<I>(get_all_dimensions_of(t));
          }(t)...);
# else
        constexpr auto c_integral = []{
          if constexpr (std::is_integral_v<decltype(c)>) return c;
          else return static_cast<std::intmax_t>(c);
        }();

        if constexpr (are_within_tolerance(c, static_cast<Scalar>(c_integral)))
        {
          return make_constant_matrix_like<T, c>(
            [](const T& t){
              if constexpr (index == I)
                return Dimensions<1>{};
              else
                return std::get<I>(get_all_dimensions_of(t));
            }(t)...);
        }
        else
        {
          return make_self_contained(c * make_constant_matrix_like<T, 1>(
            [](const T& t){
              if constexpr (index == I)
                return Dimensions<1>{};
              else
                return std::get<I>(get_all_dimensions_of(t));
            }(t)...));
        }
#endif
      }
      else
      {
        auto c = calc_const_reduce_constant_runtime(runtime_dimension_of<index>(t), b, constant_coefficient_v<T>);

        return c * make_constant_matrix_like<T, 1>(
          [](const T& t){
            if constexpr (index == I)
              return Dimensions<1>{};
            else
              return std::get<I>(get_all_dimensions_of(t));
          }(t)...);
      }
    }


    template<typename BinaryFunction, typename Arg>
    constexpr decltype(auto)
    reduce_impl(const BinaryFunction&, Arg&& arg)
    {
      return std::forward<Arg>(arg);
    }


    template<std::size_t index, std::size_t...indices, typename BinaryFunction, typename Arg>
    constexpr decltype(auto)
    reduce_impl(const BinaryFunction& b, Arg&& arg)
    {
      using Scalar = scalar_type_of_t<Arg>;

      if constexpr (constant_matrix<Arg>)
      {
        auto red = const_reduce<Scalar, index>(
          b, std::forward<Arg>(arg), std::make_index_sequence<max_indices_of_v<Arg>> {});
        return reduce_impl<indices...>(b, std::move(red));
      }
      /*else if constexpr (constant_diagonal_matrix<Arg>)
      {
        auto red = const_diag_reduce<Scalar, index>(
          b, std::forward<Arg>(arg), std::make_index_sequence<max_indices_of_v<Arg>> {});
        return reduce_impl<indices...>(b, std::move(red));
      }*/
      else
      {
        return interface::ArrayOperations<std::decay_t<Arg>>::template reduce<index, indices...>(
          b, std::forward<Arg>(arg));
      }
    }


    template<typename BinaryFunction, typename Arg, std::size_t...I>
    constexpr decltype(auto) reduce_all_indices(const BinaryFunction& b, Arg&& arg, std::index_sequence<I...>)
    {
      if constexpr (((index_dimension_of_v<Arg, I> == 1) and ...))
        return std::forward<Arg>(arg);
      else
        return reduce_impl<I...>(b, std::forward<Arg>(arg));
    }

  }


  /**
   * \brief Use a binary function to reduce a tensor across one or more of its indices.
   * \tparam indices The indices to be reduced. For example, if indices includes 0, the result will have only one row.
   * If indices includes 1, the result will have only one column. Reductions will be performed in the listed index
   * order, but unless it matters, the indices should preferably be listed in numerical order. If no indices are
   * given, the function will reduce over all indices in index order.
   * \tparam BinaryFunction A binary function invocable with two values of type <code>scalar_type_of_t<Arg></code>.
   * It must be an associative function. Preferably, it should be a constexpr function, and even more preferably,
   * it should be a standard c++ function such as std::plus or std::multiplies.
   * \tparam Arg The tensor
   */
#ifdef __cpp_concepts
  template<std::size_t...indices, typename BinaryFunction, indexible Arg> requires
    ((indices < max_indices_of_v<Arg>) and ...) and
    std::is_invocable_r_v<scalar_type_of_t<Arg>, BinaryFunction&&, scalar_type_of_t<Arg>>
#else
  template<std::size_t...indices, typename BinaryFunction, typename Arg, std::enable_if_t<indexible<Arg> and
    ((indices < max_indices_of<Arg>::value) and ...) and
    std::is_invocable<typename scalar_type_of<Arg>::type, BinaryFunction&&,
      typename scalar_type_of<Arg>::type>::value, int> = 0>
#endif
  constexpr decltype(auto)
  reduce(const BinaryFunction& b, Arg&& arg)
  {
    if constexpr (sizeof...(indices) == 0)
    {
      return detail::reduce_all_indices(b, std::forward<Arg>(arg), std::make_index_sequence<max_indices_of_v<Arg>> {});
    }
    else if constexpr (((index_dimension_of_v<Arg, indices> == 1) and ...))
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return detail::reduce_impl<indices...>(b, std::forward<Arg>(arg));
    }
  }


  /// Create a column vector by taking the mean of each row in a set of column vectors. \todo Is this necessary?
#ifdef __cpp_concepts
  template<indexible Arg>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    if constexpr (index_dimension_of_v<Arg, 1> == 1)
      return std::forward<Arg>(arg);
    else
      return make_self_contained(reduce<1>(std::plus<void>{}, std::forward<Arg>(arg)) / runtime_dimension_of<1>(arg));
  }


  /// Create a row vector by taking the mean of each column in a set of row vectors. \todo Is this necessary?
#ifdef __cpp_concepts
  template<indexible Arg>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_rows(Arg&& arg) noexcept
  {
    if constexpr (index_dimension_of_v<Arg, 0> == 1)
      return std::forward<Arg>(arg);
    else
      return make_self_contained(reduce<0>(std::plus<void>{}, std::forward<Arg>(arg)) / runtime_dimension_of<0>(arg));
  }


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
#else
  template<ElementOrder order = ElementOrder::column_major, typename BinaryFunction, typename Accum, typename Arg,
    std::enable_if_t<std::is_invocable<const std::remove_reference_t<Accum>&, const BinaryFunction&,
      Accum&&, scalar_type_of_t<Arg>>::value and std::is_move_constructible<std::decay_t<Accum>>::value and
    std::is_copy_constructible<std::decay_t<Accum>>::value, int> = 0>
#endif
  constexpr decltype(auto)
  fold(const BinaryFunction& b, Accum&& accum, Arg&& arg)
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
      return interface::ArrayOperations<std::decay_t<Arg>>::template fold<order>(
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

    constexpr std::size_t dim_n = dynamic_rows<Arg> ? column_dimension_of_v<Arg> : row_dimension_of_v<Arg>;

    auto dim = get_dimensions_of<dynamic_rows<Arg> ? 1 : 0>(arg);

    if constexpr (identity_matrix<Arg>)
    {
      return make_constant_matrix_like<Arg, 1>(dim, Dimensions<1>{});
    }
    else if constexpr (zero_matrix<Arg>)
    {
      if constexpr (not square_matrix<Arg>) detail::check_if_square_at_runtime(arg);
      return make_zero_matrix_like<dim_n, 1>(arg);
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
      return make_constant_matrix_like<Arg, c>(Dimensions<dim>{}, Dimensions<1>{});
#  else
      constexpr auto c_integral = []{
        if constexpr (std::is_integral_v<decltype(c)>) return c;
        else return static_cast<std::intmax_t>(c);
      }();

      if constexpr (are_within_tolerance(c, static_cast<Scalar>(c_integral)))
        return make_constant_matrix_like<Arg, c_integral>(dim, Dimensions<1>{});
      else
        return make_self_contained(c * make_constant_matrix_like<Arg, 1>(dim, Dimensions<1>{}));
#  endif
    }
    else
    {
      return interface::Conversions<std::decay_t<Arg>>::diagonal_of(std::forward<Arg>(arg));
    }
  }


  // ================================== //
  //  Modular transformation functions  //
  // ================================== //

#ifdef __cpp_concepts
  template<fixed_coefficients C, untyped_columns Arg> requires
    dynamic_rows<Arg> or equivalent_to<C, row_coefficient_types_of_t<Arg>>
#else
  template<typename C, typename Arg, std::enable_if_t<fixed_coefficients<C> and untyped_columns<Arg> and
    (dynamic_rows<Arg> or equivalent_to<C, row_coefficient_types_of_t<Arg>>), int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Arg&& arg)
  {
    if constexpr (dynamic_rows<Arg>) if (runtime_dimension_of<0>(arg) != C::dimension)
      throw std::out_of_range {"Number of rows (" + std::to_string(runtime_dimension_of<0>(arg)) +
        ") is incompatible with dimension types (" + std::to_string(C::dimension) + ")"};

    if constexpr (C::axes_only)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return interface::ModularTransformationTraits<Arg>::template to_euclidean<C>(std::forward<Arg>(arg));
    }
  }


#ifdef __cpp_concepts
  template<untyped_columns Arg>
#else
  template<typename Arg, std::enable_if_t<untyped_columns<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Arg&& arg)
  {
    return to_euclidean<row_coefficient_types_of_t<Arg>, Arg>(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<indexible Arg, dynamic_coefficients C>
#else
  template<typename C, typename Arg, std::enable_if_t<indexible<Arg> and dynamic_coefficients<C>, int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Arg&& arg, C&& c) noexcept
  {
    if (runtime_dimension_of<0>(arg) != c.runtime_dimension)
      throw std::out_of_range {"Number of rows (" + std::to_string(runtime_dimension_of<0>(arg)) +
        ") is incompatible with dimension types (" + std::to_string(c.runtime_dimension) + ")"};

    return interface::ModularTransformationTraits<Arg>::to_euclidean(std::forward<Arg>(arg), std::forward<C>(c));
  }


#ifdef __cpp_concepts
  template<fixed_coefficients C, untyped_columns Arg> requires
    dynamic_rows<Arg> or equivalent_to<C, row_coefficient_types_of_t<Arg>>
#else
  template<typename C, typename Arg, std::enable_if_t<fixed_coefficients<C> and untyped_columns<Arg> and
    (dynamic_rows<Arg> or equivalent_to<C, row_coefficient_types_of_t<Arg>>), int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Arg&& arg) noexcept
  {
    if constexpr (dynamic_rows<Arg>) if (runtime_dimension_of<0>(arg) != C::euclidean_dimension)
      throw std::out_of_range {"Number of rows (" + std::to_string(runtime_dimension_of<0>(arg)) +
        ") is incompatible with dimension types (" + std::to_string(C::euclidean_dimension) + ")"};

    if constexpr (C::axes_only)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return interface::ModularTransformationTraits<Arg>::template from_euclidean<C>(std::forward<Arg>(arg));
    }
  }


#ifdef __cpp_concepts
  template<untyped_columns Arg>
#else
  template<typename Arg, std::enable_if_t<untyped_columns<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Arg&& arg)
  {
    return from_euclidean<row_coefficient_types_of_t<Arg>, Arg>(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<indexible Arg, dynamic_coefficients C>
#else
  template<typename C, typename Arg, std::enable_if_t<indexible<Arg> and dynamic_coefficients<C>, int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Arg&& arg, C&& c) noexcept
  {
    if (runtime_dimension_of<0>(arg) != c.runtime_euclidean_dimension)
      throw std::out_of_range {"Number of rows (" + std::to_string(runtime_dimension_of<0>(arg)) +
        ") is incompatible with dimension types (" + std::to_string(c.runtime_euclidean_dimension) + ")"};

    return interface::ModularTransformationTraits<Arg>::from_euclidean(std::forward<Arg>(arg), std::forward<C>(c));
  }


#ifdef __cpp_concepts
  template<fixed_coefficients C, untyped_columns Arg> requires
    dynamic_rows<Arg> or equivalent_to<C, row_coefficient_types_of_t<Arg>>
#else
  template<typename C, typename Arg, std::enable_if_t<fixed_coefficients<C> and untyped_columns<Arg> and
    (dynamic_rows<Arg> or equivalent_to<C, row_coefficient_types_of_t<Arg>>), int> = 0>
#endif
  constexpr decltype(auto)
  wrap_angles(Arg&& arg)
  {
    if constexpr (dynamic_rows<Arg>) if (runtime_dimension_of<0>(arg) != C::dimension)
      throw std::out_of_range {"Number of rows (" + std::to_string(runtime_dimension_of<0>(arg)) +
        ") is incompatible with dimension types (" + std::to_string(C::dimension) + ")"};

    if constexpr (C::axes_only or identity_matrix<Arg> or zero_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      interface::ModularTransformationTraits<Arg>::template wrap_angles<C>(std::forward<Arg>(arg));
    }
  }


#ifdef __cpp_concepts
  template<untyped_columns Arg>
#else
  template<typename Arg, std::enable_if_t<untyped_columns<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  wrap_angles(Arg&& arg)
  {
    return wrap_angles<row_coefficient_types_of_t<Arg>, Arg>(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<indexible Arg, dynamic_coefficients C>
#else
  template<typename C, typename Arg, std::enable_if_t<indexible<Arg> and dynamic_coefficients<C>, int> = 0>
#endif
  constexpr decltype(auto)
  wrap_angles(Arg&& arg, C&& c) noexcept
  {
    if (runtime_dimension_of<0>(arg) != c.runtime_dimension)
      throw std::out_of_range {"Number of rows (" + std::to_string(runtime_dimension_of<0>(arg)) +
        ") is incompatible with dimension types (" + std::to_string(c.runtime_dimension) + ")"};

    return interface::ModularTransformationTraits<Arg>::wrap_angles(std::forward<Arg>(arg), std::forward<C>(c));
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
      return get_element(std::forward<Arg>(arg), std::size_t(0), std::size_t(0));
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


  namespace detail
  {
    template<typename A, typename B>
    void solve_check_A_and_B_rows_match(const A& a, const B& b)
    {
      if (runtime_dimension_of<0>(a) != runtime_dimension_of<0>(b))
        throw std::domain_error {"The rows of the two operands of the solve function must be the same, but instead "
          "the first operand has " + std::to_string(runtime_dimension_of<0>(a)) + " rows and the second operand has " +
          std::to_string(runtime_dimension_of<0>(b)) + " rows"};
    }

    template<typename A, typename B>
    auto solve_make_zero_result(const A& a, const B& b)
    {
      constexpr auto a_cols = column_dimension_of_v<A>;
      if constexpr (a_cols == dynamic_size)
      {
        auto a_runtime_cols = runtime_dimension_of<1>(a);
        if constexpr (dynamic_columns<B>)
          return make_zero_matrix_like<B, a_cols>(a_runtime_cols, runtime_dimension_of<1>(b));
        else
          return make_zero_matrix_like<B, a_cols>(a_runtime_cols);
      }
      else
      {
        return make_zero_matrix_like<a_cols>(b);
      }
    }
  }


  /**
   * \brief Solve the equation AX = B for X, which may or may not be a unique solution.
   * \details The interface to the relevant linear algebra library determines what happens if A is not invertible.
   * \tparam must_be_unique Determines whether the function throws an exception if the solution X is non-unique
   * (e.g., if the equation is under-determined)
   * \tparam must_be_exact Determines whether the function throws an exception if it cannot return an exact solution,
   * such as if the equation is over-determined. * If <code>false<code>, then the function will return an estimate
   * instead of throwing an exception.
   * \tparam A The matrix A in the equation AX = B
   * \tparam B The matrix B in the equation AX = B
   * \return The unique solution X of the equation AX = B. If <code>must_be_unique</code>, then the function can return
   * any valid solution for X. In particular, if <code>must_be_unique</code>, the function has the following behavior:
   * - If A is a \ref zero_matrix, then the result X will also be a \ref zero_matrix
   */
  #ifdef __cpp_concepts
  template<bool must_be_unique = false, bool must_be_exact = false, typename A, typename B> requires
    (dynamic_rows<A> or dynamic_rows<B> or row_dimension_of_v<A> == row_dimension_of_v<B>) and
    (not zero_matrix<A> or not zero_matrix<B> or not must_be_unique) and
    (not zero_matrix<A> or not (constant_matrix<B> or constant_diagonal_matrix<B>) or zero_matrix<B> or not must_be_exact) and
    (not constant_matrix<A> or not constant_diagonal_matrix<B> or any_dynamic_dimension<A> or
      (row_dimension_of_v<A> <= column_dimension_of_v<A> and row_dimension_of_v<B> <= column_dimension_of_v<A>) or
      (row_dimension_of_v<A> == 1 and row_dimension_of_v<B> == 1) or not must_be_exact)
  #else
  template<bool must_be_unique = false, bool must_be_exact = false, typename A, typename B, std::enable_if_t<
    (dynamic_rows<A> or dynamic_rows<B> or row_dimension_of_v<A> == row_dimension_of_v<B>) and
    (not zero_matrix<A> or not zero_matrix<B> or not must_be_unique) and
    (not zero_matrix<A> or not (constant_matrix<B> or constant_diagonal_matrix<B>) or zero_matrix<B> or not must_be_exact) and
    (not constant_matrix<A> or not constant_diagonal_matrix<B> or any_dynamic_dimension<A> or
      (row_dimension_of_v<A> <= column_dimension_of_v<A> and row_dimension_of_v<B> <= column_dimension_of_v<A>) or
      (row_dimension_of_v<A> == 1 and row_dimension_of_v<B> == 1) or not must_be_exact), int> = 0>
  #endif
  constexpr auto
  solve(A&& a, B&& b)
  {
    using Scalar = scalar_type_of_t<A>;

    if constexpr (zero_matrix<B>)
    {
      if constexpr (dynamic_rows<A> or dynamic_rows<B>) detail::solve_check_A_and_B_rows_match(a, b);

      if constexpr (must_be_unique and not constant_matrix<A> and not constant_diagonal_matrix<A>)
      {
        if (a == make_zero_matrix_like(a))
          throw std::runtime_error {"solve function requires a unique solution, "
            "but because operands A and B are both zero matrices, result X may take on any value"};
        else
          return detail::solve_make_zero_result(a, b);
      }
      else
        return detail::solve_make_zero_result(a, b);
    }
    else if constexpr (zero_matrix<A>) //< This will be a non-exact solution unless b is zero.
    {
      if constexpr (dynamic_rows<A> or dynamic_rows<B>) detail::solve_check_A_and_B_rows_match(a, b);
      return detail::solve_make_zero_result(a, b);
    }
    else if constexpr (constant_diagonal_matrix<A>)
    {
      if constexpr (dynamic_rows<A> or dynamic_rows<B>) detail::solve_check_A_and_B_rows_match(a, b);
      if constexpr (identity_matrix<A>)
        return std::forward<B>(b);
      else
        return make_self_contained(std::forward<B>(b) / constant_diagonal_coefficient_v<A>);
    }
    else if constexpr (constant_matrix<A>)
    {
      constexpr auto a_const = constant_coefficient_v<A>;

      if constexpr ((row_dimension_of_v<A> == 1 or row_dimension_of_v<B> == 1) and column_dimension_of_v<A> == 1)
      {
        if constexpr (dynamic_rows<A> or dynamic_rows<B>) detail::solve_check_A_and_B_rows_match(a, b);
        return make_self_contained(std::forward<B>(b) / a_const);
      }
      else if constexpr (constant_matrix<B>)
      {
        if constexpr (dynamic_rows<A> or dynamic_rows<B>) detail::solve_check_A_and_B_rows_match(a, b);

        constexpr auto b_const = constant_coefficient_v<B>;
        constexpr auto a_cols = column_dimension_of_v<A>;

        if constexpr (a_cols == dynamic_size)
        {
          auto a_runtime_cols = runtime_dimension_of<1>(a);
          auto c = static_cast<Scalar>(b_const) / (a_runtime_cols * a_const);
          return make_self_contained(c * make_constant_matrix_like<B, 1>(Dimensions{a_runtime_cols}, get_dimensions_of<1>(b)));
        }
        else
        {
  #if __cpp_nontype_template_args >= 201911L
          constexpr auto c = static_cast<Scalar>(b_const) / (a_cols * a_const);
          return make_constant_matrix_like<B, c>(Dimensions<a_cols>{}, get_dimensions_of<1>(b));
  #else
          if constexpr(b_const % (a_cols * a_const) == 0)
          {
            constexpr std::size_t c = static_cast<std::size_t>(b_const) / (a_cols * static_cast<std::size_t>(a_const));
            return make_constant_matrix_like<B, c>(Dimensions<a_cols>{}, get_dimensions_of<1>(b));
          }
          else
          {
            auto c = static_cast<Scalar>(b_const) / (a_cols * a_const);
            return make_self_contained(c * make_constant_matrix_like<B, 1>(Dimensions<a_cols>{}, get_dimensions_of<1>(b)));
          }
  #endif
        }
      }
      else if constexpr (row_dimension_of_v<A> == 1 or row_dimension_of_v<B> == 1 or
        (not must_be_exact and (not must_be_unique or
          (not any_dynamic_dimension<A> and row_dimension_of_v<A> >= column_dimension_of_v<A>))))
      {
        if constexpr (dynamic_rows<A> or dynamic_rows<B>) detail::solve_check_A_and_B_rows_match(a, b);
        return make_self_contained(b / (runtime_dimension_of<1>(a) * constant_coefficient_v<A>));
      }
      else //< The solution will be non-exact unless every row of b is identical.
      {
        return interface::LinearAlgebra<std::decay_t<A>>::template solve<must_be_unique, must_be_exact>(
          std::forward<A>(a), std::forward<B>(b));
      }
    }
    else if constexpr (diagonal_matrix<A> or
      ((row_dimension_of_v<A> == 1 or row_dimension_of_v<B> == 1) and column_dimension_of_v<A> == 1))
    {
      auto op = [](auto&& b_elem, auto&& a_elem) {
        if (a_elem == 0)
        {
          using Scalar = scalar_type_of_t<B>;
          if constexpr (not std::numeric_limits<Scalar>::has_infinity) throw std::logic_error {
            "In solve function, an element should be infinite, but the scalar type does not have infinite values"};
          else return std::numeric_limits<Scalar>::infinity();
        }
        else
        {
          return std::forward<decltype(b_elem)>(b_elem) / std::forward<decltype(a_elem)>(a_elem);
        }
      };
      return n_ary_operation_with_broadcasting<B>(
        get_all_dimensions_of(b), std::move(op), std::forward<B>(b), diagonal_of(std::forward<A>(a)));
    }
    else
    {
      return interface::LinearAlgebra<std::decay_t<A>>::template solve<must_be_unique, must_be_exact>(
        std::forward<A>(a), std::forward<B>(b));
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
