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

  // --------------------- //
  //  get_tensor_order_of  //
  // --------------------- //

  namespace detail
  {
    template<std::size_t...I, typename T>
    constexpr auto get_tensor_order_of_impl(std::index_sequence<I...>, const T& t)
    {
      return ((IndexTraits<T, I>::dimension_at_runtime(t) == 1 ? 0 : 1) + ... + 0);
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
    if constexpr (not has_dynamic_dimensions<T>)
      return tensor_order_of_v<T>;
    else
      return detail::get_tensor_order_of_impl(std::make_index_sequence<max_indices_of_v<T>> {}, t);
  }


  // -------------------------- //
  //   get_index_dimension_of   //
  // -------------------------- //

  /**
   * \brief Get the runtime dimensions of index N of \ref indexible T
   */
#ifdef __cpp_concepts
  template<std::size_t N, indexible T>
#else
  template<std::size_t N, typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr std::size_t
  get_index_dimension_of(const T& t)
  {
    constexpr auto dim = index_dimension_of_v<T, N>;
    if constexpr (dim == dynamic_size) return IndexTraits<T, N>::dimension_at_runtime(t);
    else return dim;
  }


  // ------------------- //
  //  get_dimensions_of  //
  // ------------------- //

#ifdef __cpp_concepts
  template<std::size_t N = 0, indexible Arg> requires (N < max_indices_of_v<Arg>) and
    (euclidean_index_descriptor<coefficient_types_of_t<Arg, N>> or
      requires(const Arg& arg) { interface::CoordinateSystemTraits<Arg, N>::coordinate_system_types_at_runtime(arg); })
#else
  template<std::size_t N = 0, typename Arg, std::enable_if_t<indexible<Arg> and N < max_indices_of<Arg>::value, int> = 0>
#endif
  constexpr auto get_dimensions_of(const Arg& arg)
  {
    using T = coefficient_types_of_t<Arg, N>;
    if constexpr (euclidean_index_descriptor<T>)
    {
      if constexpr (dynamic_dimension<Arg, N>)
        return Dimensions{interface::IndexTraits<std::decay_t<Arg>, N>::dimension_at_runtime(arg)};
      else
        return Dimensions<index_dimension_of_v<Arg, N>> {};
    }
    else
    {
      if constexpr (dynamic_dimension<Arg, N>)
        return interface::CoordinateSystemTraits<Arg, N>::coordinate_system_types_at_runtime(arg);
      else
        return coefficient_types_of_t<Arg, N> {};
    }
  }


  // ----------------------- //
  //  get_all_dimensions_of  //
  // ----------------------- //

  namespace detail
  {
    template<typename T, std::size_t...I>
    constexpr auto get_all_dimensions_of_impl(const T& t, std::index_sequence<I...>)
    {
      return std::tuple {get_dimensions_of<I>(t)...};
    }


    template<typename T, std::size_t...I>
    constexpr auto get_all_dimensions_of_impl(std::index_sequence<I...>)
    {
      return std::tuple {coefficient_types_of_t<T, I> {}...};
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
  constexpr decltype(auto) get_all_dimensions_of(const T& t)
  {
    return detail::get_all_dimensions_of_impl(t, std::make_index_sequence<max_indices_of_v<T>> {});
  }


  /**
   * \overload
   * \brief Return a tuple of \ref index_descriptor objects defining the dimensions of T.
   * \details This overload is only enabled if all dimensions of T are known at compile time.
   * \tparam T A matrix or array
   */
#ifdef __cpp_concepts
  template<indexible T> requires (not has_dynamic_dimensions<T>)
#else
  template<typename T, std::enable_if_t<indexible<T> and not has_dynamic_dimensions<T>, int> = 0>
#endif
  constexpr auto get_all_dimensions_of()
  {
    return detail::get_all_dimensions_of_impl<T>(std::make_index_sequence<max_indices_of_v<T>> {});
  }


  // ----------------------------------------- //
  //  make_default_dense_writable_matrix_like  //
  // ----------------------------------------- //

  /**
   * \brief Make a default, dense, writable matrix based on a list of Dimensions describing the sizes of each index.
   * \description If
   * \tparam T A dummy matrix or array from the relevant library (size and shape does not matter)
   * \param d a tuple of Dimensions describing the sizes of each index. This can be omitted if T is of fixed size.
   * In that case, the index descriptors will be derived from T.
   */
#ifdef __cpp_concepts
  template<indexible T, scalar_type Scalar = scalar_type_of_t<T>, index_descriptor...D> requires
    (sizeof...(D) == max_indices_of_v<T>) or (sizeof...(D) == 0 and not has_dynamic_dimensions<T>)
#else
  template<typename T, typename Scalar = scalar_type_of_t<T>, typename...D, std::enable_if_t<
    indexible<T> and scalar_type<Scalar> and (index_descriptor<D> and ...) and
    (sizeof...(D) == max_indices_of_v<T> or (sizeof...(D) == 0 and not has_dynamic_dimensions<T>)), int> = 0>
#endif
  constexpr decltype(auto)
  make_default_dense_writable_matrix_like(D&&...d)
  {
    if constexpr (sizeof...(D) == 0)
      return std::apply(
        [](auto&&...d) { return make_default_dense_writable_matrix_like<T, Scalar>(std::forward<decltype(d)>(d)...); },
        get_all_dimensions_of<T>());
    else
      return EquivalentDenseWritableMatrix<std::decay_t<T>, std::decay_t<Scalar>>::make_default(std::forward<D>(d)...);
  }


  /**
   * \overload
   * \brief Make a default, dense, writable matrix based on an existing object.
   * \param t The existing object
   * \tparam Scalar An optional scalar type for the new matrix. By default, t's scalar type is used.
   */
#ifdef __cpp_concepts
  template<typename Scalar = void, indexible T>
#else
  template<typename Scalar = void, typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr auto
  make_default_dense_writable_matrix_like(const T& t)
  {
    using NewScalar = std::conditional_t<std::is_void_v<Scalar>, scalar_type_of_t<T>, Scalar>;
    return std::apply(
      [](auto&&...d) { return make_default_dense_writable_matrix_like<T, NewScalar>(std::forward<decltype(d)>(d)...); },
      get_all_dimensions_of(t));
  }


  // --------------------------------- //
  //  make_dense_writable_matrix_from  //
  // --------------------------------- //

  /**
   * \brief Convert the argument to a dense, writable matrix of a type based on native matrix T.
   * \tparam Arg The object from which the new matrix is based
   * \tparam Scalar The Scalar type of the new matrix, if different than that of Arg
   */
#ifdef __cpp_concepts
  template<indexible Arg, scalar_type Scalar = scalar_type_of_t<Arg>>
#else
  template<typename Arg, typename Scalar = scalar_type_of_t<Arg>, std::enable_if_t<
    indexible<Arg> and scalar_type<Scalar>, int> = 0>
#endif
  constexpr decltype(auto)
  make_dense_writable_matrix_from(Arg&& arg) noexcept
  {
    using Trait = EquivalentDenseWritableMatrix<std::decay_t<Arg>, std::decay_t<Scalar>>;
    using Nat = std::remove_reference_t<decltype(Trait::convert(std::declval<Arg&&>()))>;

    static_assert(not std::is_const_v<Nat>, "EquivalentDenseWritableMatrix::convert must not return a const result");

    if constexpr (std::is_same_v<std::decay_t<Arg>, Nat>)
      return std::forward<Arg>(arg);
    else
      return Trait::convert(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Create a dense, writable matrix with size and shape based on M, filled with a set of scalar components
   * \tparam M The matrix or array on which the new matrix is patterned.
   * \tparam Scalar An optional scalar type for the new matrix. By default, M's scalar type is used.
   * \tparam Ds Index descriptors describing the size of the resulting object.
   * \param d_tup A tuple of index descriptors Ds
   * \param args Scalar values to fill the new matrix.
   */
#ifdef __cpp_concepts
  template<indexible M, scalar_type Scalar = scalar_type_of_t<M>, index_descriptor...Ds, std::convertible_to<const Scalar> ... Args>
  requires (sizeof...(Args) % ((dynamic_index_descriptor<Ds> ? 1 : dimension_size_of_v<Ds>) * ... * 1) == 0)
#else
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wdiv-by-zero"
  template<typename M, typename Scalar = scalar_type_of_t<M>, typename...Ds, typename...Args, std::enable_if_t<
    indexible<M> and scalar_type<Scalar> and (index_descriptor<Ds> and ...) and
    (std::is_convertible_v<Args, const Scalar> and ...) and
    (sizeof...(Args) % ((dynamic_index_descriptor<Ds> ? 1 : dimension_size_of_v<Ds>) * ... * 1) == 0), int> = 0>
#endif
  inline auto
  make_dense_writable_matrix_from(const std::tuple<Ds...>& d_tup, Args ... args)
  {
    using Trait = EquivalentDenseWritableMatrix<std::decay_t<M>, std::decay_t<Scalar>>;
    using Nat = decltype(Trait::make_default(std::declval<Ds&&>()...));
    return MatrixTraits<Nat>::make(static_cast<const Scalar>(args)...);
  }
#ifndef __cpp_concepts
# pragma GCC diagnostic pop
#endif


  namespace detail
  {
    template<typename M, std::size_t...I>
    constexpr auto count_fixed_dims(std::index_sequence<I...>)
    {
      return ((dynamic_dimension<M, I> ? 1 : index_dimension_of_v<M, I>) * ... * 1);
    }


    template<typename M, std::size_t N>
    constexpr auto check_make_dense_args()
    {
      constexpr auto dims = count_fixed_dims<M>(std::make_index_sequence<max_indices_of_v<M>> {});
      return (N % dims == 0) and number_of_dynamic_indices_v<M> <= 1;
    }


    template<typename M, std::size_t dims, typename Scalar, std::size_t...I, typename...Args>
    inline auto make_dense_writable_matrix_from_impl(std::index_sequence<I...>, Args...args)
    {
      constexpr auto d = sizeof...(Args) / dims;
      return make_dense_writable_matrix_from<M, Scalar>(
        std::tuple {[]{
          if constexpr(dynamic_dimension<M, I>) return Dimensions<d>{};
          else return coefficient_types_of_t<M, I> {};
        }()...}, args...);
    }

  } // namespace detail


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
  template<indexible M, typename Scalar = scalar_type_of_t<M>, std::convertible_to<const Scalar> ... Args>
  requires (detail::check_make_dense_args<M, sizeof...(Args)>())
#else
  template<typename M, typename Scalar = scalar_type_of_t<M>, typename ... Args, std::enable_if_t<
    indexible<M> and (std::is_convertible_v<Args, const Scalar> and ...) and
    (detail::check_make_dense_args<M, sizeof...(Args)>()), int> = 0>
#endif
  inline auto
  make_dense_writable_matrix_from(Args ... args)
  {
    constexpr std::make_index_sequence<max_indices_of_v<M>> seq;
    constexpr auto dims = detail::count_fixed_dims<M>(seq);
    return detail::make_dense_writable_matrix_from_impl<M, dims, Scalar>(seq, args...);
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

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename Scalar, typename = void, typename...D>
    struct make_zero_matrix_trait_defined: std::false_type {};

    template<typename T, typename Scalar, typename...D>
    struct make_zero_matrix_trait_defined<T, Scalar, std::void_t<
      decltype(SingleConstantMatrixTraits<T, Scalar>::make_zero_matrix(std::declval<D&&>()...))>, D...>
      : std::true_type {};
  }
#endif


  /**
   * \brief Make a zero matrix.
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   * \tparam Scalar An optional scalar type for the new zero matrix. By default, T's scalar type is used.
   * \param D A set of \ref index_descriptor "index descriptors" defining the dimensions of each index.
   */
#ifdef __cpp_concepts
  template<indexible T, typename Scalar = scalar_type_of_t<T>, index_descriptor...D> requires
    (sizeof...(D) == max_indices_of_v<T>)
#else
  template<typename T, typename Scalar = scalar_type_of_t<T>, typename...D,
    std::enable_if_t<indexible<T> and (index_descriptor<D> and ...) and sizeof...(D) == max_indices_of_v<T>, int> = 0>
#endif
  constexpr decltype(auto)
  make_zero_matrix_like(D&&...d)
  {
    using Td = std::decay_t<T>;
#ifdef __cpp_concepts
    if constexpr (requires (D&&...d) { SingleConstantMatrixTraits<T, Scalar>::make_zero_matrix(std::forward<D>(d)...); })
#else
    if constexpr (detail::make_zero_matrix_trait_defined<Td, Scalar, void, D...>::value)
#endif
    {
      return SingleConstantMatrixTraits<Td, Scalar>::make_zero_matrix(std::forward<D>(d)...);
    }
    else
    {
      // Default behavior if interface function not defined:
      using N = std::decay_t<decltype(EquivalentDenseWritableMatrix<Td, std::decay_t<Scalar>>::make_default(std::declval<D&&>()...))>;
      return Eigen3::ZeroMatrix<N> {std::forward<D>(d)...};
    }
  }


  /**
   * \overload
   * \brief Make a zero matrix based on the argument, specifying a new scalar type.
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   * \tparam Scalar A scalar type for the new matrix.
   */
#ifdef __cpp_concepts
  template<typename Scalar, indexible T>
#else
  template<typename Scalar, typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr decltype(auto)
  make_zero_matrix_like(T&& t)
  {
    if constexpr (zero_matrix<T> and std::is_same_v<Scalar, scalar_type_of_t<T>>)
      return std::forward<T>(t);
    else
      return std::apply(
        [](auto&&...arg){ return make_zero_matrix_like<T, Scalar>(std::forward<decltype(arg)>(arg)...); },
        get_all_dimensions_of(t));
  }


  /**
   * \overload
   * \brief Make a zero matrix based on T.
   * \tparam T The matrix or array on which the new matrix is patterned.
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr decltype(auto)
  make_zero_matrix_like(T&& t)
  {
    return make_zero_matrix_like<scalar_type_of_t<T>>(std::forward<T>(t));
  }


  /**
   * \overload
   * \brief Make a zero matrix based on the type T, where the dimensions of T are known at compile time.
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   * \tparam Scalar A scalar type for the new matrix (by default, the same as that of T)
   */
#ifdef __cpp_concepts
  template<indexible T, typename Scalar = scalar_type_of_t<T>> requires (not has_dynamic_dimensions<T>)
#else
  template<typename T, typename Scalar = scalar_type_of_t<T>, std::enable_if_t<
    indexible<T> and (not has_dynamic_dimensions<T>), int> = 0>
#endif
  constexpr decltype(auto)
  make_zero_matrix_like()
  {
    return std::apply(
      [](auto&&...arg){ return make_zero_matrix_like<T, Scalar>(std::forward<decltype(arg)>(arg)...); },
      get_all_dimensions_of<T>());
  }


  // --------------------------- //
  //  make_constant_matrix_like  //
  // --------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename Scalar, auto constant, typename = void, typename...D>
    struct make_constant_matrix_trait_defined: std::false_type {};

    template<typename T, typename Scalar, auto constant, typename...D>
    struct make_constant_matrix_trait_defined<T, Scalar, constant, std::void_t<
      decltype(SingleConstantMatrixTraits<T, Scalar>::template make_constant_matrix<constant>(std::declval<D&&>()...))>, D...>
      : std::true_type {};
  }
#endif


  /**
   * \brief Make a constant matrix with size and shape modeled at least partially on T
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   * \tparam constant The constant
   * \tparam Scalar An optional scalar type for the new zero matrix. By default, T's scalar type is used.
   * \param D A set of \ref index_descriptor "index descriptors" defining the dimensions of each index.
   * \todo Swap T and constant positions
   */
#ifdef __cpp_concepts
  template<indexible T, auto constant, typename Scalar = scalar_type_of_t<T>, index_descriptor...D> requires
    (sizeof...(D) == max_indices_of_v<T>)
#else
  template<typename T, auto constant, typename Scalar = scalar_type_of_t<T>, typename...D,
    std::enable_if_t<indexible<T> and (index_descriptor<D> and ...) and sizeof...(D) == max_indices_of_v<T>, int> = 0>
#endif
  constexpr decltype(auto)
  make_constant_matrix_like(D&&...d)
  {
    using Td = std::decay_t<T>;
#ifdef __cpp_concepts
    if constexpr (requires (D&&...d) {
      SingleConstantMatrixTraits<std::decay_t<T>, std::decay_t<Scalar>>::template make_constant_matrix<constant>(std::forward<D>(d)...);
    })
#else
    if constexpr (detail::make_zero_matrix_trait_defined<Td, Scalar, void, D...>::value)
#endif
    {
      return SingleConstantMatrixTraits<std::decay_t<T>, std::decay_t<Scalar>>::template make_constant_matrix<constant>(std::forward<D>(d)...);
    }
    else
    {
      // Default behavior if interface function not defined:
      using N = std::decay_t<decltype(EquivalentDenseWritableMatrix<Td, std::decay_t<Scalar>>::make_default(std::declval<D&&>()...))>;
      return Eigen3::ConstantMatrix<N, constant> {std::forward<D>(d)...};
    }
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

    if constexpr (constants_match and std::is_same_v<Scalar, scalar_type_of_t<T>>)
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


  /**
   * \overload
   * \brief Make a single-constant matrix based on the type T, where the dimensions of T are known at compile time.
   * \tparam T The matrix or array on which the new constant matrix is patterned.
   * \tparam constant The constant.
   * \tparam Scalar A scalar type for the new matrix (by default, the same as that of T)
   * \todo Swap T and constant positions
   */
#ifdef __cpp_concepts
  template<indexible T, auto constant, typename Scalar = scalar_type_of_t<T>> requires (not has_dynamic_dimensions<T>)
#else
  template<typename T, auto constant, typename Scalar = scalar_type_of_t<T>, std::enable_if_t<
    indexible<T> and (not has_dynamic_dimensions<T>), int> = 0>
#endif
  constexpr decltype(auto)
  make_constant_matrix_like()
  {
    return std::apply(
      [](auto&&...arg){ return make_constant_matrix_like<T, constant, Scalar>(std::forward<decltype(arg)>(arg)...); },
      get_all_dimensions_of<T>());
  }


  // --------------------------- //
  //  make_identity_matrix_like  //
  // --------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename Scalar, typename D, typename = void>
    struct make_identity_matrix_trait_defined: std::false_type {};

    template<typename T, typename Scalar, typename D>
    struct make_identity_matrix_trait_defined<T, Scalar, D, std::void_t<
      decltype(SingleConstantDiagonalMatrixTraits<T, Scalar>::make_identity_matrix(std::declval<D&&>()))>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Make an identity matrix.
   * \tparam T The matrix or array on which the identity matrix is patterned.
   * \tparam Scalar An optional scalar type for the new zero matrix. By default, T's scalar type is used.
   * \param D An \ref index_descriptor "index descriptor" defining the dimensions of each index.
   */
#ifdef __cpp_concepts
  template<indexible T, typename Scalar = scalar_type_of_t<T>, index_descriptor D>
#else
  template<typename T, typename Scalar = typename scalar_type_of<T>::type, typename D, std::enable_if_t<
    indexible<T> and index_descriptor<D>, int> = 0>
#endif
  constexpr auto
  make_identity_matrix_like(D&& d)
  {
    using Td = std::decay_t<T>;
#ifdef __cpp_concepts
    if constexpr (requires (D&& d) { SingleConstantDiagonalMatrixTraits<Td, Scalar>::make_identity_matrix(std::forward<D>(d)); })
#else
    if constexpr (detail::make_zero_matrix_trait_defined<Td, Scalar, D>::value)
#endif
    {
      return SingleConstantDiagonalMatrixTraits<Td, Scalar>::make_identity_matrix(std::forward<D>(d));
    }
    else
    {
      // Default behavior if interface function not defined:
      return Eigen3::DiagonalMatrix {make_constant_matrix_like<Td, 1, Scalar>(std::forward<D>(d), Dimensions<1>{})};
    }

  }


  /**
   * \overload
   * \brief Make an identity matrix based on the argument, specifying a new scalar type.
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   * \tparam Scalar A scalar type for the new matrix.
   */
#ifdef __cpp_concepts
  template<typename Scalar, indexible T> requires has_dynamic_dimensions<T> or square_matrix<T>
#else
  template<typename Scalar, typename T, std::enable_if_t<indexible<T> and
    (has_dynamic_dimensions<T> or square_matrix<T>), int> = 0>
#endif
  constexpr decltype(auto)
  make_identity_matrix_like(T&& t)
  {
    if constexpr (identity_matrix<T> and std::is_same_v<Scalar, scalar_type_of_t<T>>)
    {
      return std::forward<T>(t);
    }
    else if constexpr (has_dynamic_dimensions<T>)
    {
      if (get_index_dimension_of<0>(t) != get_index_dimension_of<1>(t)) throw std::invalid_argument {
        "Argument of make_identity_matrix_like must be square; instead it has " +
        std::to_string(get_index_dimension_of<0>(t)) + " rows and " +
        std::to_string(get_index_dimension_of<1>(t)) + " columns"};

      if constexpr (dynamic_dimension<T, 0>)
        return make_identity_matrix_like<T, Scalar>(get_dimensions_of<1>(t));
      else
        return make_identity_matrix_like<T, Scalar>(get_dimensions_of<0>(t));
    }
    else
    {
      return make_identity_matrix_like<T, Scalar>(get_dimensions_of<0>(t));
    }
  }


  /**
   * \overload
   * \brief Make an identity matrix based on the argument.
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   */
#ifdef __cpp_concepts
  template<indexible T> requires has_dynamic_dimensions<T> or square_matrix<T>
#else
  template<typename T, std::enable_if_t<indexible<T> and (has_dynamic_dimensions<T> or square_matrix<T>), int> = 0>
#endif
  constexpr decltype(auto)
  make_identity_matrix_like(T&& t)
  {
    return make_identity_matrix_like<scalar_type_of_t<T>>(std::forward<T>(t));
  }


  /**
   * \overload
   * \brief Make an identity matrix based on T, which has fixed size, specifying a new scalar type.
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   * \tparam Scalar A scalar type for the new matrix. The default is the scalar type of T.
   */
#ifdef __cpp_concepts
  template<indexible T, typename Scalar = scalar_type_of_t<T>> requires square_matrix<T>
#else
  template<typename T, typename Scalar = scalar_type_of_t<T>, std::enable_if_t<indexible<T> and square_matrix<T>, int> = 0>
#endif
  constexpr decltype(auto)
  make_identity_matrix_like()
  {
    return make_identity_matrix_like<T, Scalar>(Dimensions<index_dimension_of_v<T, 0>>{});
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

    template<typename T, std::size_t...I>
    constexpr bool no_recursive_runtime_parameters(std::index_sequence<I...>)
    {
      return ((not Dependencies<T>::has_runtime_parameters) and ... and
        no_recursive_runtime_parameters<std::decay_t<std::tuple_element_t<I, typename Dependencies<T>::type>>>(
          std::make_index_sequence<std::tuple_size_v<typename Dependencies<std::decay_t<std::tuple_element_t<I, typename Dependencies<T>::type>>>::type>> {}
          ));
    }

#ifdef __cpp_concepts
    template<typename T>
    concept all_lvalue_ref_dependencies =
      no_recursive_runtime_parameters<std::decay_t<T>>(
        std::make_index_sequence<std::tuple_size_v<typename Dependencies<std::decay_t<T>>::type>> {}) and
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
    constexpr void check_index_bounds(const Arg& arg, std::index_sequence<seq...>, I...i)
    {
      if constexpr (sizeof...(I) == 1)
      {
        std::size_t dim_i = (i,...);
        auto c = get_index_dimension_of<1>(arg);
        if (c == 1)
        {
          auto r = get_index_dimension_of<0>(arg);
          if (dim_i >= r)
            throw std::out_of_range {((std::string {set ? "s" : "g"} + "et_element:") + " Row index is " +
            std::to_string(dim_i) + " but should be in range [0..." + std::to_string(r-1) + "].")};
        }
        else
        {
          if (dim_i >= c)
            throw std::out_of_range {((std::string {set ? "s" : "g"} + "et_element:") + " Column index is " +
            std::to_string(dim_i) + " but should be range [0..." + std::to_string(c-1) + "].")};
        }
      }
      else
      {
        (((i >= get_index_dimension_of<seq>(arg)) ?
          throw std::out_of_range {(("At least one " + std::string {set ? "s" : "g"} +
            "et_element index out of range:") + ... + (" Index " + std::to_string(seq) + " is " +
            std::to_string(i) + " and should be in range [0..." +
            std::to_string(get_index_dimension_of<seq>(arg)-1) + "]."))} :
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
    if constexpr (constant_matrix<Arg>)
    {
      return constant_coefficient_v<Arg>;
    }
    else
    {
      detail::check_index_bounds<false>(arg, std::make_index_sequence<sizeof...(I)> {}, i...);
      return interface::GetElement<std::decay_t<Arg>, I...>::get(std::forward<Arg>(arg), i...);
    }
  }
#else
  template<typename Arg, typename...I, std::enable_if_t<(std::is_convertible_v<I, std::size_t> and ...) and
    element_gettable<Arg, std::conditional_t<std::is_same_v<I, std::size_t>, I, std::size_t>...> and
    (sizeof...(I) != 1 or column_vector<Arg> or row_vector<Arg>), int> = 0>
  constexpr decltype(auto) get_element(Arg&& arg, const I...i)
  {
    detail::check_index_bounds<false>(arg, std::make_index_sequence<sizeof...(I)> {}, i...);
    return interface::GetElement<std::decay_t<Arg>, void, I...>::get(std::forward<Arg>(arg), i...);
  }
#endif


  /// Set element to s using I... indices.
#ifdef __cpp_concepts
  template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>&> Scalar, std::convertible_to<std::size_t>...I>
    requires element_settable<Arg&, std::conditional_t<std::same_as<I, std::size_t>, I, std::size_t>...>
  inline void set_element(Arg& arg, Scalar s, const I...i)
  {
    detail::check_index_bounds<true>(arg, std::make_index_sequence<sizeof...(I)> {}, i...);
    return interface::SetElement<std::decay_t<Arg>, I...>::set(arg, s, i...);
  }
#else
  template<typename Arg, typename Scalar, typename...I, std::enable_if_t<
    (std::is_convertible_v<I, std::size_t> and ...) and
    std::is_convertible_v<Scalar, const scalar_type_of_t<Arg>&> and
    element_settable<Arg&, std::conditional_t<std::is_same_v<I, std::size_t>, I, std::size_t>...>, int> = 0>
  inline void set_element(Arg& arg, Scalar s, const I...i)
  {
    detail::check_index_bounds<true>(arg, std::make_index_sequence<sizeof...(I)> {}, i...);
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
      auto cols = get_index_dimension_of<1>(arg);

      if (runtime_i >= cols) throw std::out_of_range {"Runtime column index (which is " + std::to_string(runtime_i) +
          ") is not in range 0 <= i < " + std::to_string(cols) + "."};

      if constexpr (zero_matrix<Arg>)
        return make_zero_matrix_like<Arg>(get_dimensions_of<0>(arg), Dimensions<1>{});
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
      auto rows = get_index_dimension_of<0>(arg);

      if (runtime_i >= rows) throw std::out_of_range {"Runtime row index (which is " + std::to_string(runtime_i) +
          ") is not in range 0 <= i < " + std::to_string(rows) + "."};

      if constexpr (zero_matrix<Arg>)
        return make_zero_matrix_like<Arg>(Dimensions<1>{}, get_dimensions_of<1>(arg));
      else
        return make_constant_matrix_like<Arg, constant_coefficient_v<Arg>>(Dimensions<1>{}, get_dimensions_of<1>(arg));
    }
    else
    {
      return interface::Subsets<std::decay_t<Arg>>::template row<compile_time_index...>(std::forward<Arg>(arg), i...);
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
      if constexpr (dynamic_columns<Arg>)
        if (get_index_dimension_of<1>(arg) != 1) throw std::domain_error {
          "Argument of to_diagonal must be a column vector, not a row vector"};
      return std::forward<Arg>(arg);
    }
    else if constexpr (zero_matrix<Arg> and dim != dynamic_size)
    {
      // note, the interface function should deal with a zero matrix of uncertain size.

      if constexpr (dynamic_columns<Arg>)
        if (get_index_dimension_of<1>(arg) != 1) throw std::domain_error {
          "Argument of to_diagonal must have 1 column; instead it has " +
          std::to_string(get_index_dimension_of<1>(arg))};
      return make_zero_matrix_like<Arg>(Dimensions<dim>{}, Dimensions<dim>{});
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
      if (get_dimensions_of<0>(arg) != get_dimensions_of<1>(arg))
        throw std::invalid_argument {"Argument of diagonal_of must be a square matrix; instead, " +
        (get_index_dimension_of<0>(arg) == get_index_dimension_of<1>(arg) ?
          "the row and column indices have non-equivalent types" :
          "it has " + std::to_string(get_index_dimension_of<0>(arg)) + " rows and " +
            std::to_string(get_index_dimension_of<1>(arg)) + "columns")};
    };
  }


  /**
   * \brief Extract the diagonal from a square matrix.
   * \tparam Arg A diagonal matrix
   * \returns Arg A column vector
   */
#ifdef __cpp_concepts
  template<typename Arg> requires (has_dynamic_dimensions<Arg> or square_matrix<Arg>)
#else
  template<typename Arg, std::enable_if_t<has_dynamic_dimensions<Arg> or square_matrix<Arg>, int> = 0>
#endif
  inline decltype(auto)
  diagonal_of(Arg&& arg)
  {
    using Scalar = scalar_type_of_t<Arg>;

    auto dim = get_dimensions_of<dynamic_rows<Arg> ? 1 : 0>(arg);

    if constexpr (one_by_one_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (identity_matrix<Arg>)
    {
      return make_constant_matrix_like<Arg, 1>(dim, Dimensions<1>{});
    }
    else if constexpr (zero_matrix<Arg>)
    {
      if constexpr (not square_matrix<Arg>) detail::check_if_square_at_runtime(arg);
      return make_zero_matrix_like<Arg>(dim, Dimensions<1>{});
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
      return make_constant_matrix_like<Arg, c>(dim, Dimensions<1>{});
#  else
      constexpr auto c_integral = static_cast<std::intmax_t>(c);
      if constexpr (are_within_tolerance(c, static_cast<Scalar>(c_integral)))
        return make_constant_matrix_like<Arg, c_integral>(dim, Dimensions<1>{});
      else
        return make_self_contained(c * to_native_matrix<Arg>(make_constant_matrix_like<Arg, 1>(dim, Dimensions<1>{})));
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
  template<indexible Arg, index_descriptor C> requires (dynamic_columns<Arg> or has_untyped_index<Arg, 1>) and
    (dynamic_index_descriptor<C> or dynamic_rows<Arg> or has_untyped_index<Arg, 0> or
      equivalent_to<C, coefficient_types_of_t<Arg, 0>>)
#else
  template<typename Arg, typename C, std::enable_if_t<index_descriptor<C> and indexible<Arg> and
    (dynamic_columns<Arg> or has_untyped_index<Arg, 1>) and
    (dynamic_index_descriptor<C> or dynamic_rows<Arg> or has_untyped_index<Arg, 0> or
      equivalent_to<C, coefficient_types_of_t<Arg, 0>>), int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Arg&& arg, const C& c) noexcept
  {
    if constexpr (euclidean_index_descriptor<C>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      if constexpr (dynamic_columns<Arg>) if (not get_index_descriptor_is_untyped(get_dimensions_of<1>(arg)))
        throw std::domain_error {"In to_euclidean, the column index is not untyped"};

      if constexpr (dynamic_rows<Arg>)
        if (not get_index_descriptor_is_untyped(get_dimensions_of<0>(arg)) and get_dimensions_of<0>(arg) != C{})
          throw std::domain_error {"In to_euclidean, the row index is not untyped and does not match the designated"
            "fixed_index_descriptor"};

      return interface::ModularTransformationTraits<Arg>::to_euclidean(std::forward<Arg>(arg), c);
    }
  }


#ifdef __cpp_concepts
  template<indexible Arg> requires (not has_untyped_index<Arg, 0>) and has_untyped_index<Arg, 1>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg> and (not has_untyped_index<Arg, 0>) and
    has_untyped_index<Arg, 1>, int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Arg&& arg)
  {
    return to_euclidean(std::forward<Arg>(arg), get_dimensions_of<0>(arg));
  }


#ifdef __cpp_concepts
  template<indexible Arg, index_descriptor C> requires (dynamic_columns<Arg> or has_untyped_index<Arg, 1>) and
    (dynamic_index_descriptor<C> or dynamic_rows<Arg> or has_untyped_index<Arg, 0> or
      equivalent_to<C, coefficient_types_of_t<Arg, 0>>)
#else
  template<typename Arg, typename C, std::enable_if_t<index_descriptor<C> and indexible<Arg> and
    (dynamic_columns<Arg> or has_untyped_index<Arg, 1>) and
    (dynamic_index_descriptor<C> or dynamic_rows<Arg> or has_untyped_index<Arg, 0> or
      equivalent_to<C, coefficient_types_of_t<Arg, 0>>), int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Arg&& arg, const C& c) noexcept
  {
    if constexpr (euclidean_index_descriptor<C>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      if constexpr (dynamic_columns<Arg>) if (not get_index_descriptor_is_untyped(get_dimensions_of<1>(arg)))
        throw std::domain_error {"In from_euclidean, the column index is not untyped"};

      if constexpr (dynamic_rows<Arg>)
        if (not get_index_descriptor_is_untyped(get_dimensions_of<0>(arg)) and get_dimensions_of<0>(arg) != C{})
          throw std::domain_error {"In from_euclidean, the row index is not untyped and does not match the designated"
            "fixed_index_descriptor"};

      return interface::ModularTransformationTraits<Arg>::from_euclidean(std::forward<Arg>(arg), c);
    }
  }


#ifdef __cpp_concepts
  template<indexible Arg> requires (not has_untyped_index<Arg, 0>) and has_untyped_index<Arg, 1>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg> and (not has_untyped_index<Arg, 0>) and
    has_untyped_index<Arg, 1>, int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Arg&& arg)
  {
    return from_euclidean(std::forward<Arg>(arg), get_dimensions_of<0>(arg));
  }


#ifdef __cpp_concepts
  template<indexible Arg, index_descriptor C> requires (dynamic_columns<Arg> or has_untyped_index<Arg, 1>) and
    (dynamic_index_descriptor<C> or dynamic_rows<Arg> or has_untyped_index<Arg, 0> or
      equivalent_to<C, coefficient_types_of_t<Arg, 0>>)
#else
  template<typename Arg, typename C, std::enable_if_t<index_descriptor<C> and indexible<Arg> and
    (dynamic_columns<Arg> or has_untyped_index<Arg, 1>) and
    (dynamic_index_descriptor<C> or dynamic_rows<Arg> or has_untyped_index<Arg, 0> or
      equivalent_to<C, coefficient_types_of_t<Arg, 0>>), int> = 0>
#endif
  constexpr decltype(auto)
  wrap_angles(Arg&& arg, const C& c)
  {
    if constexpr (euclidean_index_descriptor<C> or identity_matrix<Arg> or zero_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      if constexpr (dynamic_columns<Arg>) if (not get_index_descriptor_is_untyped(get_dimensions_of<1>(arg)))
        throw std::domain_error {"In wrap_angles, the column index is not untyped"};

      if constexpr (dynamic_rows<Arg>)
        if (not get_index_descriptor_is_untyped(get_dimensions_of<0>(arg)) and get_dimensions_of<0>(arg) != C{})
          throw std::domain_error {"In wrap_angles, the row index is not untyped and does not match the designated"
            "fixed_index_descriptor"};

      interface::ModularTransformationTraits<Arg>::wrap_angles(std::forward<Arg>(arg), c);
    }
  }


#ifdef __cpp_concepts
  template<indexible Arg> requires (not has_untyped_index<Arg, 0>) and has_untyped_index<Arg, 1>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg> and (not has_untyped_index<Arg, 0>) and
    has_untyped_index<Arg, 1>, int> = 0>
#endif
  constexpr decltype(auto)
  wrap_angles(Arg&& arg)
  {
    return wrap_angles(std::forward<Arg>(arg), get_dimensions_of<0>(arg));
  }


  // ================== //
  //  Array operations  //
  // ================== //

  // ----------------- //
  //  n_ary_operation  //
  // ----------------- //

  namespace detail
  {
    template<typename T>
    struct is_plus : std::false_type {};

    template<typename T>
    struct is_plus<std::plus<T>> : std::true_type {};

    template<typename T>
    struct is_multiplies : std::false_type {};

    template<typename T>
    struct is_multiplies<std::multiplies<T>> : std::true_type {};


#ifdef __cpp_concepts
    template<typename Op, typename...Scalar>
#else
    template<typename Op, typename = void, typename...Scalar>
#endif
    struct is_constexpr_n_ary_function_impl : std::false_type {};


#ifdef __cpp_concepts
    template<typename Op, typename...Scalar> requires requires(Scalar...x) { Op{}(x...); }
    struct is_constexpr_n_ary_function_impl<Op, Scalar...>
#else
    template<typename Op, typename...Scalar>
    struct is_constexpr_n_ary_function_impl<Op, std::void_t<decltype(Op{}(std::declval<Scalar>()...))>, Scalar...>
#endif
      : std::true_type {};


    template<typename Op, typename...Scalar>
#ifdef __cpp_concepts
    struct is_constexpr_n_ary_function : is_constexpr_n_ary_function_impl<Op, Scalar...> {};
#else
    struct is_constexpr_n_ary_function : is_constexpr_n_ary_function_impl<Op, void, Scalar...> {};
#endif


    template<typename Operation, typename...Args, std::size_t...I>
    constexpr bool is_invocable_with_indices(std::index_sequence<I...>)
    {
      return std::is_invocable_v<Operation&&, Args&&..., decltype(I)...>;
    }


  template<typename Operation, std::size_t...I, typename...Args>
  constexpr decltype(auto) n_ary_invoke_op(Operation&& operation, std::index_sequence<I...> seq, Args&&...args)
  {
    if constexpr (is_invocable_with_indices<Operation, Args...>(seq))
      return std::forward<Operation>(operation)(std::forward<Args>(args)..., static_cast<decltype(I)>(0)...);
    else
      return std::forward<Operation>(operation)(std::forward<Args>(args)...);
  }


#ifdef __cpp_concepts
  template<typename Operation, std::size_t indices, typename...Args>
#else
  template<typename Operation, std::size_t indices, typename = void, typename...Args>
#endif
  struct n_ary_operator_traits_impl {};


#ifdef __cpp_concepts
  template<typename Operation, std::size_t indices, typename...Args>
  requires (is_invocable_with_indices<Operation, Args...>(std::make_index_sequence<indices> {})) or
    std::is_invocable_v<Operation&&, Args&&...>
  struct n_ary_operator_traits_impl<Operation, indices, Args...>
#else
  template<typename Operation, std::size_t indices, typename...Args>
  struct n_ary_operator_traits_impl<Operation, indices, std::enable_if_t<
    is_invocable_with_indices<Operation, Args...>(std::make_index_sequence<indices> {}) or
    std::is_invocable_v<Operation&&, Args&&...>>, Args...>
#endif
  {
    using type = decltype(n_ary_invoke_op(
      std::declval<Operation&&>(), std::make_index_sequence<indices> {}, std::declval<Args&&>()...));
  };


  template<typename Operation, std::size_t indices, typename...Args>
  struct n_ary_operator_traits
#ifdef __cpp_concepts
    : n_ary_operator_traits_impl<Operation, indices, Args...> {};
#else
    : n_ary_operator_traits_impl<Operation, indices, void, Args...> {};
#endif


#ifndef __cpp_concepts
    template<typename Operation, std::size_t Indices, typename = void, typename...Args>
    struct n_ary_operator_impl : std::false_type {};

    template<typename Operation, std::size_t Indices, typename...Args>
    struct n_ary_operator_impl<Operation, Indices, std::enable_if_t<(indexible<Args> and ...) and
      (std::is_invocable<Operation&&, typename std::add_lvalue_reference<typename scalar_type_of<Args>::type>::type...>::value or
        is_invocable_with_indices<Operation, typename std::add_lvalue_reference<typename scalar_type_of<Args>::type>::type...>(
          std::make_index_sequence<Indices> {}))>, Args...>
    : std::true_type {};
#endif


    template<typename Operation, std::size_t Indices, typename...Args>
#ifdef __cpp_concepts
    concept n_ary_operator = (indexible<Args> and ...) and
      (std::is_invocable_v<Operation&&, std::add_lvalue_reference_t<scalar_type_of_t<Args>>...> or
        is_invocable_with_indices<Operation, std::add_lvalue_reference_t<scalar_type_of_t<Args>>...>(
          std::make_index_sequence<Indices> {}));
#else
    constexpr bool n_ary_operator = n_ary_operator_impl<Operation, Indices, void, Args...>::value;
#endif


#ifdef __cpp_concepts
    template<typename T, typename DTup, typename Op, typename...Args>
#else
    template<typename T, typename DTup, typename Op, typename = void, typename...Args>
#endif
    struct interface_defines_n_ary_operation : std::false_type {};


    template<typename T, typename DTup, typename Op, typename...Args>
#ifdef __cpp_concepts
    requires requires(const DTup& d_tup, Op op, Args...args) {
      interface::ArrayOperations<std::decay_t<T>>::n_ary_operation(d_tup, op, std::forward<Args>(args)...);
    }
    struct interface_defines_n_ary_operation<T, DTup, Op, Args...>
#else
    struct interface_defines_n_ary_operation<T, DTup, Op, std::void_t<
      decltype(interface::ArrayOperations<std::decay_t<T>>::n_ary_operation_with_indices(
        std::declval<const DTup&>(), std::declval<Op>(), std::declval<Args>()...))>, Args...>
#endif
      : std::true_type {};


#ifdef __cpp_concepts
    template<typename T, typename DTup, typename Op, typename...Args>
#else
    template<typename T, typename DTup, typename Op, typename = void, typename...Args>
#endif
    struct interface_defines_n_ary_operation_with_indices : std::false_type {};


    template<typename T, typename DTup, typename Op, typename...Args>
#ifdef __cpp_concepts
    requires requires(const DTup& d_tup, Op op, Args...args) {
      interface::ArrayOperations<std::decay_t<T>>::n_ary_operation_with_indices(d_tup, op, std::forward<Args>(args)...);
    }
    struct interface_defines_n_ary_operation_with_indices<T, DTup, Op, Args...>
#else
    struct interface_defines_n_ary_operation_with_indices<T, DTup, Op, std::void_t<
      decltype(interface::ArrayOperations<std::decay_t<T>>::n_ary_operation_with_indices(
        std::declval<const DTup&>(), std::declval<Op>(), std::declval<Args>()...))>, Args...>
#endif
      : std::true_type {};


    template<typename Arg, std::size_t...I, typename...J>
    inline decltype(auto) n_ary_operation_get_element_impl(Arg&& arg, std::index_sequence<I...>, J...j)
    {
      if constexpr (sizeof...(I) == sizeof...(J))
        return get_element(std::forward<Arg>(arg), (j < get_index_dimension_of<I>(arg) ? j : 0)...);
      else
        return get_element(std::forward<Arg>(arg), [](auto dim, const auto& j_tup){
          auto j = std::get<I>(j_tup);
          if (j < dim) return j;
          else return 0;
        }(get_index_dimension_of<I>(arg), std::tuple {j...})...);
    }


    template<typename Operation, typename ArgsTup, std::size_t...ArgI, typename...J>
    inline auto n_ary_operation_get_element(Operation&& operation, ArgsTup&& args_tup, std::index_sequence<ArgI...>, J...j)
    {
      if constexpr (std::is_invocable_v<Operation&&, scalar_type_of_t<std::tuple_element_t<ArgI, std::decay_t<ArgsTup>>>..., J...>)
        return std::forward<Operation>(operation)(n_ary_operation_get_element_impl(
          std::get<ArgI>(std::forward<ArgsTup>(args_tup)),
          std::make_index_sequence<max_indices_of_v<std::tuple_element_t<ArgI, std::decay_t<ArgsTup>>>> {},
          j...)..., j...);
      else
        return std::forward<Operation>(operation)(n_ary_operation_get_element_impl(
          std::get<ArgI>(std::forward<ArgsTup>(args_tup)),
          std::make_index_sequence<max_indices_of_v<std::tuple_element_t<ArgI, std::decay_t<ArgsTup>>>> {},
          j...)...);
    }


    template<typename M, typename Operation, typename ArgsTup, typename...J>
    inline void n_ary_operation_iterate(M& m, Operation&& operation, ArgsTup&& args_tup, std::index_sequence<>, J...j)
    {
      set_element(m, n_ary_operation_get_element(std::forward<Operation>(operation), args_tup,
        std::make_index_sequence<std::tuple_size_v<ArgsTup>> {}, j...), j...);
    }


    template<typename M, typename Operation, typename ArgsTup, std::size_t I, std::size_t...Is, typename...J>
    inline void n_ary_operation_iterate(M& m, Operation&& operation, ArgsTup&& args_tup, std::index_sequence<I, Is...>, J...j)
    {
      for (std::size_t i = 0; i < get_index_dimension_of<I>(m); i++)
        n_ary_operation_iterate(m, operation, std::forward<ArgsTup>(args_tup), std::index_sequence<Is...> {}, j..., i);
    }


    template<typename PatternMatrix, typename...Ds, typename Op, typename...Args>
    static constexpr decltype(auto)
    n_ary_operation_with_broadcasting_impl(const std::tuple<Ds...>& tup, Op&& op, Args&&...args)
    {
      // zero_matrix:
      if constexpr (sizeof...(Args) > 0 and (zero_matrix<Args> and ...) and
        (is_plus<Op>::value or is_multiplies<Op>::value))
      {
        using Scalar = decltype(op(std::forward<Args>(args)...));
        return std::apply(
          [](auto&&...ds){ return make_zero_matrix_like<PatternMatrix, Scalar>(std::forward<decltype(ds)>(ds)...); },
          tup);
      }

      // constant_matrix:
      else if constexpr (sizeof...(Args) > 0 and (constant_matrix<Args> and ...) and
        is_constexpr_n_ary_function<Op, scalar_type_of_t<Args>...>::value)
      {

        constexpr auto c = Op{}(constant_coefficient_v<Args>...);
        using Scalar = std::decay_t<decltype(c)>;
# if __cpp_nontype_template_args >= 201911L
        return std::apply(
          [](auto&&...ds){ return make_constant_matrix_like<PatternMatrix, c, Scalar>(std::forward<decltype(ds)>(ds)...); },
          tup);
# else
        constexpr auto c_integral = static_cast<std::intmax_t>(c);
        if constexpr (are_within_tolerance(c, static_cast<Scalar>(c_integral)))
          return std::apply(
            [](auto&&...ds){
              return make_constant_matrix_like<PatternMatrix, c_integral, Scalar>(std::forward<decltype(ds)>(ds)...);
            },
            tup);
        else
          return make_self_contained(c * to_native_matrix<PatternMatrix>(std::apply(
            [](auto&&...ds){
              return make_constant_matrix_like<PatternMatrix, 1, Scalar>(std::forward<decltype(ds)>(ds)...);
            },
            tup)));
# endif
      }

      // other cases:
      else
      {
        constexpr std::make_index_sequence<max_indices_of_v<PatternMatrix>> seq;
        if constexpr (is_invocable_with_indices<Op&&, std::add_lvalue_reference_t<scalar_type_of_t<Args>>...>(seq) and
          detail::interface_defines_n_ary_operation_with_indices<PatternMatrix, std::tuple<Ds...>, Op&&, Args&&...>::value)
        {
          using Trait = interface::ArrayOperations<std::decay_t<PatternMatrix>>;
          return Trait::n_ary_operation_with_indices(tup, std::forward<Op>(op), std::forward<Args>(args)...);
        }
        else if constexpr (is_invocable_with_indices<Op&&, std::add_lvalue_reference_t<scalar_type_of_t<Args>>...>(seq) and
          detail::interface_defines_n_ary_operation<PatternMatrix, std::tuple<Ds...>, Op&&, Args&&...>::value)
        {
          using Trait = interface::ArrayOperations<std::decay_t<PatternMatrix>>;
          return Trait::n_ary_operation(tup, std::forward<Op>(op), std::forward<Args>(args)...);
        }
        else
        {
          using Scalar = std::decay_t<typename n_ary_operator_traits<Op, max_indices_of_v<PatternMatrix>,
            std::add_lvalue_reference_t<scalar_type_of_t<Args>>...>::type>;
          auto m = std::apply(
            [](auto&&...ds){
              return make_default_dense_writable_matrix_like<PatternMatrix, Scalar>(std::forward<decltype(ds)>(ds)...);
            },
            tup);
          n_ary_operation_iterate(m, std::forward<Op>(op), std::forward_as_tuple(std::forward<Args>(args)...), seq);
          return m;
        }
      }
    }


    template<typename...Ds, typename Op, typename Arg, typename...Args>
    static constexpr auto
    n_ary_operation_with_broadcasting_impl(const std::tuple<Ds...>& tup, Op&& op, Arg&& arg, Args&&...args)
    {
      return n_ary_operation_with_broadcasting_impl<Arg>(tup, std::forward<Op>(op), std::forward<Arg>(arg), std::forward<Args>(args)...);
    }


#ifdef __cpp_concepts
    template<typename DTup, typename Arg, std::size_t...indices>
#else
    template<typename DTup, typename Arg, typename = void, std::size_t...indices>
#endif
    struct n_ary_argument_index : std::false_type {};


#ifdef __cpp_concepts
    template<typename DTup, typename Arg, std::size_t...indices>
    requires (max_indices_of_v<Arg> <= sizeof...(indices)) or
      ((dimension_size_of_v<std::tuple_element_t<indices, DTup>> == dynamic_size or
      index_dimension_of_v<Arg, indices> == dynamic_size or
      (index_dimension_of_v<Arg, indices> == 1 and euclidean_index_descriptor<coefficient_types_of_t<Arg, indices>>) or
      equivalent_to<coefficient_types_of_t<Arg, indices>, std::tuple_element_t<indices, DTup>> or
      (index_dimension_of_v<Arg, indices> == dimension_size_of_v<std::tuple_element_t<indices, DTup>> and
        euclidean_index_descriptor<coefficient_types_of_t<Arg, indices>>)) and ...)
    struct n_ary_argument_index<DTup, Arg, indices...>
#else
    template<typename DTup, typename Arg, std::size_t...indices>
    struct n_ary_argument_index<DTup, Arg, std::enable_if_t<max_indices_of_v<Arg> <= sizeof...(indices)>, indices...>
    : std::true_type {};

    template<typename DTup, typename Arg, std::size_t...indices>
    struct n_ary_argument_index<DTup, Arg, std::enable_if_t<(max_indices_of_v<Arg> > sizeof...(indices)) and
      ((dimension_size_of_v<std::tuple_element_t<indices, DTup>> == dynamic_size or
      index_dimension_of_v<Arg, indices> == dynamic_size or
      (index_dimension_of_v<Arg, indices> == 1 and euclidean_index_descriptor<coefficient_types_of_t<Arg, indices>>) or
      equivalent_to<coefficient_types_of_t<Arg, indices>, std::tuple_element_t<indices, DTup>> or
      (index_dimension_of_v<Arg, indices> == dimension_size_of_v<std::tuple_element_t<indices, DTup>> and
        euclidean_index_descriptor<coefficient_types_of_t<Arg, indices>>)) and ...)>, indices...>
#endif
    : std::true_type {};


    template<typename DTup, typename Arg, std::size_t...indices>
    constexpr bool n_ary_argument_impl(std::index_sequence<indices...>)
    {
# ifdef __cpp_concepts
      return n_ary_argument_index<DTup, Arg, indices...>::value;
# else
      return n_ary_argument_index<DTup, Arg, void, indices...>::value;
# endif
    }


    // Arg is a valid argument to n_ary_operation
    template<typename Arg, typename...Ds>
#ifdef __cpp_concepts
    concept n_ary_argument =
#else
    constexpr bool n_ary_argument =
#endif
      indexible<Arg> and (n_ary_argument_impl<std::tuple<Ds...>, Arg>(std::make_index_sequence<sizeof...(Ds)> {}));


    template<typename...Ds, typename Arg, std::size_t...indices>
    inline void check_n_ary_rt_dims_impl(const std::tuple<Ds...>& d_tup, const Arg& arg, std::index_sequence<indices...>)
    {
      ((get_index_dimension_of<indices>(arg) == 1 or
        get_index_dimension_of<indices>(arg) == get_dimension_size_of(std::get<indices>(d_tup)) ? 0 : throw std::logic_error {
          "In an argument to n_ary_operation, the dimension of index " + std::to_string(indices) +
          " is " + std::to_string(get_index_dimension_of<indices>(arg)) + ", but should be 1 " +
          (get_dimension_size_of(std::get<indices>(d_tup)) == 1 ? "" : "or " +
          std::to_string(get_dimension_size_of(std::get<indices>(d_tup)))) +
          "(the dimension of index " + std::to_string(indices) + " of the PatternMatrix template argument)"}),...);
    }


    // Check that runtime dimensions of arguments Args are compatible with index descriptors Ds.
    template<typename...Ds, typename...Arg>
    inline void check_n_ary_runtime_dimensions(const std::tuple<Ds...>& d_tup, const Arg&...arg)
    {
      (check_n_ary_rt_dims_impl(d_tup, arg, std::make_index_sequence<sizeof...(Ds)> {}), ...);
    }

  } // namespace detail


  /**
   * \brief Perform a component-wise n-ary operation, using broadcasting to match the size of a pattern matrix.
   * \details This overload is for unary, binary, and higher n-ary operations. Examples:
   * - Unary operation, no broadcasting:
   *   \code
   *     auto ds32 = std::tuple {Dimensions<3>{}, Dimensions<2>{}};
   *     auto op1 = [](auto arg){return 3 * arg;};
   *     auto M = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
   *     auto m32 = make_dense_writable_matrix_from<M>(ds32, 1, 2, 3, 4, 5, 6);
   *     std::cout << n_ary_operation(ds32, op1, m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     3, 6,
   *     9, 12,
   *     15, 18
   *   \endcode
   * - Unary operation, broadcasting:
   *   \code
   *     auto ds31 = std::tuple {Dimensions<3>{}, Dimensions<1>{}};
   *     auto m31 = make_dense_writable_matrix_from<M>(ds31, 1, 2, 3);
   *     std::cout << n_ary_operation(ds32, op1, m31) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     3, 3,
   *     6, 6,
   *     9, 9
   *   \endcode
   * - Binary operation, no broadcasting:
   *   \code
   *     auto op2 = [](auto arg1, auto arg2){return 3 * arg1 + arg2;};
   *     std::cout << n_ary_operation(ds32, op2, m32, 2 * m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     5, 10,
   *     15, 20,
   *     25, 30
   *   \endcode
   * - Binary operation, broadcasting:
   *   \code
   *     std::cout << n_ary_operation(ds32, op2, m31, 2 * m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     5, 7,
   *     12, 14,
   *     19, 21
   *   \endcode
   * - Binary operation, broadcasting, with indices:
   *   \code
   *     auto op2b = [](auto arg1, auto arg2, std::size_t row, std::size_t col){return 3 * arg1 + arg2 + row + col;};
   *     std::cout << n_ary_operation(ds32, op2b, m31, 2 * m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     5, 8,
   *     13, 16,
   *     21, 24
   *   \endcode
   * \tparam Ds Index descriptors defining the size of the result
   * \tparam Operation The n-ary operation taking n arguments and, optionally, a set of indices indicating the location
   * within the result. The number of indices, if any, must match the number of indices in the result.
   * \tparam Args The arguments
   * \return A matrix or array in which each component is the result of calling Operation on corresponding components
   * from each of the arguments, in the order specified.
   */
#ifdef __cpp_concepts
  template<index_descriptor...Ds, typename Operation, detail::n_ary_argument<Ds...>...Args>
  requires (sizeof...(Args) > 0) and detail::n_ary_operator<Operation, sizeof...(Ds), Args...>
#else
  template<typename...Ds, typename Operation, typename...Args, std::enable_if_t<(sizeof...(Args) > 0) and
    (index_descriptor<Ds> and ...) and detail::n_ary_operator<Operation, sizeof...(Ds), Args...> and
    (detail::n_ary_argument<Args, Ds...> and ...), int> = 0>
#endif
  constexpr decltype(auto)
  n_ary_operation(const std::tuple<Ds...>& d_tup, Operation&& operation, Args&&...args)
  {
    if constexpr (((dimension_size_of_v<Ds> == dynamic_size) or ...) or (has_dynamic_dimensions<Args> or ...))
      detail::check_n_ary_runtime_dimensions(d_tup, args...);

    return detail::n_ary_operation_with_broadcasting_impl(d_tup, std::forward<Operation>(operation), std::forward<Args>(args)...);
  }


  namespace detail
  {
    template<std::size_t I, typename Arg, typename...Args>
    constexpr auto find_max_runtime_dims_impl(const Arg& arg, const Args&...args)
    {
      if constexpr (sizeof...(Args) == 0)
      {
        return get_index_dimension_of<I>(arg);
      }
      else
      {
        auto dim0 = get_dimension_size_of(get_dimensions_of<I>(arg));
        auto dim = get_dimension_size_of(find_max_runtime_dims_impl<I>(args...));

        if (dim0 == dim or dim == 1) return dim0;
        else if (dim0 == 1) return dim;
        else throw std::logic_error {"In an argument to n_ary_operation, the dimension of index " +
          std::to_string(I) + " is " + std::to_string(dim0) + ", which is not 1 and does not match index " +
          std::to_string(I) + " of a later argument, which is " + std::to_string(dim)};
      }
    }


    template<std::size_t I, typename...Args>
    constexpr auto find_max_dims_impl(const Args&...args)
    {
      constexpr auto max_stat_dim = std::max({(dynamic_dimension<Args, I> ? 0 : index_dimension_of_v<Args, I>)...});
      constexpr auto dim = max_stat_dim == 0 ? dynamic_size : max_stat_dim;

      if constexpr (((not dynamic_dimension<Args, I> and (index_dimension_of_v<Args, I> == 0 or
          (index_dimension_of_v<Args, I> != 1 and index_dimension_of_v<Args, I> != dim))) or ...))
        throw std::logic_error {"The dimension of arguments to n_ary_operation should be either "
          "1 or the maximum dimension among the arguments. Instead, the argument dimensions are" +
          ((" " + std::to_string(index_dimension_of_v<Args, I>) + " (index " + std::to_string(I) + ")") + ...)};

      if constexpr ((dim != dynamic_size and dim > 1) or (dim == 1 and not (dynamic_dimension<Args, I> or ...)))
        return Dimensions<dim>{};
      else
        return Dimensions<dynamic_size>{find_max_runtime_dims_impl<I>(args...)};
    }


    template<std::size_t...I, typename...Args>
    constexpr auto find_max_dims(std::index_sequence<I...>, const Args&...args)
    {
      return std::tuple {find_max_dims_impl<I>(args...)...};
    }


    template<typename Arg, std::size_t...I>
    constexpr decltype(auto) n_ary_get_element_0(Arg&& arg, std::index_sequence<I...>)
    {
      return get_element(std::forward<Arg>(arg), (I * 0)...);
    }


    //// operation_returns_lvalue_reference ////

#ifdef __cpp_concepts
    template<typename Operation, typename...Args>
#else
    template<typename Operation, typename = void, typename...Args>
#endif
    struct operation_returns_lvalue_reference_impl : std::false_type {};


#ifdef __cpp_concepts
    template<typename Operation, typename Arg>
    requires (not std::is_const_v<Arg>) and
      std::is_lvalue_reference_v<typename n_ary_operator_traits<Operation, max_indices_of_v<Arg>,
        std::add_lvalue_reference_t<scalar_type_of_t<Arg>>>::type>
    struct operation_returns_lvalue_reference_impl<Operation, Arg&>
#else
    template<typename Operation, typename Arg>
    struct operation_returns_lvalue_reference_impl<Operation, std::enable_if_t<
      (not std::is_const<Arg>::value) and
      std::is_lvalue_reference<typename n_ary_operator_traits<Operation, max_indices_of<Arg>::value,
        typename std::add_lvalue_reference<typename scalar_type_of<Arg>::type>::type>::type>::value>, Arg&>
#endif
    : std::true_type {};


    template<typename Operation, typename...Args>
#ifdef __cpp_concepts
    concept operation_returns_lvalue_reference = operation_returns_lvalue_reference_impl<Operation, Args...>::value;
#else
    constexpr bool operation_returns_lvalue_reference = operation_returns_lvalue_reference_impl<Operation, void, Args...>::value;
#endif


  //// operation_returns_void ////

#ifdef __cpp_concepts
    template<typename Operation, typename...Args>
#else
    template<typename Operation, typename = void, typename...Args>
#endif
    struct operation_returns_void_impl : std::false_type {};


#ifdef __cpp_concepts
    template<typename Operation, typename Arg>
    requires (not std::is_const_v<Arg>) and
      std::is_void_v<typename n_ary_operator_traits<Operation, max_indices_of_v<Arg>,
        std::add_lvalue_reference_t<scalar_type_of_t<Arg>>>::type>
    struct operation_returns_void_impl<Operation, Arg&>
#else
    template<typename Operation, typename Arg>
    struct operation_returns_void_impl<Operation, std::enable_if_t<
      (not std::is_const_v<Arg>) and
      std::is_void_v<typename n_ary_operator_traits<Operation, max_indices_of_v<Arg>,
        std::add_lvalue_reference_t<scalar_type_of_t<Arg>>>::type>>, Arg&>
#endif
    : std::true_type {};


    template<typename Operation, typename...Args>
#ifdef __cpp_concepts
    concept operation_returns_void = operation_returns_void_impl<Operation, Args...>::value;
#else
    constexpr bool operation_returns_void = operation_returns_void_impl<Operation, void, Args...>::value;
#endif


    template<typename Operation, typename Arg, typename...J>
    inline void unary_operation_in_place_impl(Operation&& operation, Arg& arg, std::index_sequence<>, J...j)
    {
      if constexpr (std::is_invocable_v<Operation&&, std::add_lvalue_reference_t<scalar_type_of_t<Arg>>, J...>)
        std::forward<Operation>(operation)(get_element(arg, j...), j...);
      else
        std::forward<Operation>(operation)(get_element(arg, j...));
    }


    template<typename Operation, typename Arg, std::size_t I, std::size_t...Is, typename...J>
    inline void unary_operation_in_place_impl(Operation&& operation, Arg& arg, std::index_sequence<I, Is...>, J...j)
    {
      for (std::size_t i = 0; i < get_index_dimension_of<I>(arg); i++)
      {
        unary_operation_in_place_impl(operation, arg, std::index_sequence<Is...> {}, j..., i);
      }
    }

  } // namespace detail


  /**
   * \overload
   * \brief Perform a component-wise n-ary operation, using broadcasting if necessary to make the arguments the same size.
   * \details Each of the arguments may be expanded by broadcasting. The result will derive each dimension from the
   * largest corresponding dimension among the arguments.
   * There are additional input options for unary operations: the operation may return either a scalar value, an
   * lvalue reference, or void. Examples:
   * - Binary operation, broadcasting:
   *   \code
   *     auto M = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
   *     auto op2a = [](auto arg1, auto arg2){return 3 * arg1 + arg2;};
   *     auto m31 = make_dense_writable_matrix_from<M>(ds31, 1, 2, 3);
   *     auto m32 = make_dense_writable_matrix_from<M>(ds32, 1, 2, 3, 4, 5, 6);
   *     std::cout << n_ary_operation(op2a, m31, 2 * m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     5, 7,
   *     12, 14,
   *     19, 21
   *   \endcode
   * - Binary operation, broadcasting, with indices:
   *   \code
   *     auto op2b = [](auto arg1, auto arg2, std::size_t row, std::size_t col){return 3 * arg1 + arg2 + row + col;};
   *     std::cout << n_ary_operation(op2b, m31, 2 * m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     5, 8,
   *     13, 16,
   *     21, 24
   *   \endcode
   * - Unary operation, with indices:
   *   \code
   *     auto op1a = [](auto& arg, std::size_t row, std::size_t col){return arg + row + col;};
   *     std::cout << n_ary_operation(op1a, m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     1, 3,
   *     4, 6,
   *     7, 9
   *   \endcode
   * - Unary operation, with indices, in-place operation returning lvalue reference:
   *   \code
   *     auto op1b = [](auto& arg, std::size_t row, std::size_t col){return arg += row + col;};
   *     std::cout << n_ary_operation(op1b, m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     1, 3,
   *     4, 6,
   *     7, 9
   *   \endcode
   * - Unary operation, with indices, in-place operation returning void:
   *   \code
   *     auto op1c = [](auto& arg, std::size_t row, std::size_t col){arg += row + col;};
   *     std::cout << n_ary_operation(op1c, m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     1, 3,
   *     4, 6,
   *     7, 9
   *   \endcode
   * \tparam Operation The n-ary operation taking n arguments and, optionally, a set of indices. The operation may
   * return one of the following:
   * - a scalar value;
   * - an lvalue reference to a scalar element within the argument; or
   * - void (for example, if the operation works on an lvalue reference)
   * \tparam Args The arguments
   * \return A matrix or array in which each component is the result of calling Operation on corresponding components
   * from each of the arguments, in the order specified.
   */
#ifdef __cpp_concepts
  template<typename Operation, indexible...Args> requires (sizeof...(Args) > 0) and
    detail::n_ary_operator<Operation, std::max({max_indices_of_v<Args>...}), Args...>
#else
  template<typename Operation, typename...Args, std::enable_if_t<(indexible<Args> and ... and (sizeof...(Args) > 0)) and
    detail::n_ary_operator<Operation, std::max({max_indices_of_v<Args>...}), Args...>, int> = 0>
#endif
  constexpr decltype(auto)
  n_ary_operation(Operation&& operation, Args&&...args)
  {
    if constexpr (detail::operation_returns_lvalue_reference<Operation&&, Args&&...>)
    {
      auto args_tup = std::forward_as_tuple(args...);
      auto& arg = std::get<0>(args_tup);
      using Arg = std::decay_t<decltype(arg)>;
      constexpr std::make_index_sequence<max_indices_of_v<Arg>> seq;

      using G = decltype(detail::n_ary_get_element_0(std::declval<Arg&>(), seq));
      static_assert(std::is_same_v<G, std::decay_t<G>&>, "Cannot use n_ary_operation with an operation that returns an "
        "lvalue reference unless get_element(...) returns a non-const lvalue reference.");

      n_ary_operation_iterate(arg, std::forward<Operation>(operation), std::move(args_tup), seq);
      return arg;
    }
    else if constexpr (detail::operation_returns_void<Operation&&, Args&&...>)
    {
      auto args_tup = std::forward_as_tuple(args...);
      auto& arg = std::get<0>(args_tup);
      using Arg = std::decay_t<decltype(arg)>;
      constexpr std::make_index_sequence<max_indices_of_v<Arg>> seq;

      using G = decltype(detail::n_ary_get_element_0(std::declval<Arg&>(), seq));
      static_assert(std::is_same_v<G, std::decay_t<G>&>, "Cannot use n_ary_operation with an operation that returns "
        "void unless get_element(...) returns a non-const lvalue reference.");

      detail::unary_operation_in_place_impl(std::forward<Operation>(operation), arg, seq);
      return arg;
    }
    else
    {
      constexpr auto max_indices = std::max({max_indices_of_v<Args>...});
      auto d_tup = detail::find_max_dims(std::make_index_sequence<max_indices> {}, args...);
      return n_ary_operation(std::move(d_tup), std::forward<Operation>(operation), std::forward<Args>(args)...);
    }
  }


  namespace detail
  {
    template<std::size_t I, std::size_t...indices>
    constexpr bool is_index_match()
    {
      return ((I == indices) or ...);
    }


    template<typename D_tup, std::size_t...indices, std::size_t...I>
    constexpr std::size_t count_index_dims(std::index_sequence<I...>)
    {
      return ([]{
        if constexpr (is_index_match<I, indices...>()) return dimension_size_of_v<std::tuple_element_t<I, D_tup>>;
        else return 1;
      }() * ... * 1);
    }


    template<typename T, std::size_t...indices, std::size_t...I>
    constexpr std::size_t pattern_index_dims(std::index_sequence<I...>)
    {
      return ([]{
        if constexpr (is_index_match<I, indices...>()) return index_dimension_of_v<T, I>;
        else return 1;
      }() * ... * 1);
    }


    template<typename PatternMatrix, typename Operation, typename Descriptors_tuple, typename Index_seq, typename K_seq, typename...Is>
    void nullary_set_elements(PatternMatrix& m, Operation&& op, const Descriptors_tuple&, Index_seq, K_seq, Is...is)
    {
      constexpr auto seq = std::make_index_sequence<max_indices_of_v<PatternMatrix>> {};
      if constexpr (detail::is_invocable_with_indices<Operation&&>(seq))
        set_element(m, op(is...), is...);
      else
        set_element(m, op(), is...);
    }


    template<std::size_t DsIndex, std::size_t...DsIndices, typename PatternMatrix, typename Operation,
      typename Descriptors_tuple, std::size_t...indices, std::size_t...Ks, typename...Is>
    void nullary_set_elements(PatternMatrix& m, Operation&& op, const Descriptors_tuple& ds_tup,
      std::index_sequence<indices...> index_seq, std::index_sequence<Ks...> k_seq, Is...is)
    {
      if constexpr (((DsIndex == indices) or ...))
      {
        constexpr std::size_t i = ((DsIndex == indices ? Ks : 0) + ...);
        nullary_set_elements<DsIndices...>(m, std::forward<Operation>(op), ds_tup, index_seq, k_seq, is..., i);
      }
      else
      {
        // Iterate through the dimensions of the current DsIndex and add set elements for each dimension iteratively.
        for (std::size_t i = 0; i < get_dimension_size_of(std::get<DsIndex>(ds_tup)); ++i)
        {
          nullary_set_elements<DsIndices...>(m, op, ds_tup, index_seq, k_seq, is..., i);
        }
      }
    }


    template<std::size_t CurrentOpIndex, std::size_t factor, typename PatternMatrix, typename Operations_tuple,
      typename Descriptors_tuple, typename UniqueIndicesSeq, std::size_t...AllDsIndices, typename K_seq>
    void nullary_iterate(PatternMatrix& m, Operations_tuple&& op_tup, const Descriptors_tuple& ds_tup,
      UniqueIndicesSeq unique_indices_seq, std::index_sequence<AllDsIndices...>, K_seq k_seq)
    {
      nullary_set_elements<AllDsIndices...>(m, std::get<CurrentOpIndex>(
        std::forward<Operations_tuple>(op_tup)), ds_tup, unique_indices_seq, k_seq);
    }


    template<std::size_t CurrentOpIndex, std::size_t factor, std::size_t index, std::size_t...indices,
      typename PatternMatrix, typename Operations_tuple, typename Descriptors_tuple, typename UniqueIndicesSeq, typename AllDsSeq,
      std::size_t...Ks, std::size_t...Js, typename...J_seqs>
    void nullary_iterate(PatternMatrix& m, Operations_tuple&& op_tup, const Descriptors_tuple& ds_tup,
      UniqueIndicesSeq unique_indices_seq, AllDsSeq all_ds_seq, std::index_sequence<Ks...>, std::index_sequence<Js...>,
      J_seqs...j_seqs)
    {
      constexpr std::size_t new_factor = factor / dimension_size_of_v<std::tuple_element_t<index, Descriptors_tuple>>;

      ((nullary_iterate<CurrentOpIndex + new_factor * Js, new_factor, indices...>(
        m, std::forward<Operations_tuple>(op_tup), ds_tup, unique_indices_seq, all_ds_seq,
        std::index_sequence<Ks..., Js>{}, j_seqs...)),...);
    }

  } // namespace detail


  /**
   * \overload
   * \brief Perform a component-wise nullary operation.
   * \details
   * - One operation for the entire matrix
   *   \code
   *     auto ds23 = std::tuple {Dimensions<2>{}, Dimensions<3>{}};
   *     auto M = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
   *     std::cout << n_ary_operation<M>(std::index_sequence<>{}, ds23, [](auto arg){return 7;}) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     7, 7, 7,
   *     7, 7, 7
   *   \endcode
   * - One operation for each element
   *   \code
   *     std::cout << n_ary_operation<M>(std::index_sequence<0, 1>{}, ds23, []{return 4;}, []{return 5;}, []{return 6;}, []{return 7;}, []{return 8;}, []{return 9;});
   *   \endcode
   *   Output:
   *   \code
   *     4, 5, 6,
   *     7, 8, 9
   *   \endcode
   * - One operation for each row
   *   \code
   *     auto ds23a = std::tuple {Dimensions<2>{}, Dimensions{3}};
   *     std::cout << n_ary_operation<M>(std::index_sequence<0>{}, ds23a, []{return 5;}, []{return 6;});
   *   \endcode
   *   Output:
   *   \code
   *     5, 5, 5,
   *     6, 6, 6
   *   \endcode
   * - One operation for each column
   *   \code
   *     auto ds23b = std::tuple {Dimensions{2}, Dimensions<3>{}};
   *     std::cout << n_ary_operation<M>(std::index_sequence<1>{}, ds23b, []{return 5;}, []{return 6;}, []{return 7;});
   *   \endcode
   *   Output:
   *   \code
   *     5, 6, 7,
   *     5, 6, 7
   *   \endcode
   * \tparam PatternMatrix A matrix or array corresponding to the result type. Its dimensions need not match the specified dimensions Ds
   * \tparam indices The indices, if any, for which there is a distinct distribution for each slice based on that index.
   * \tparam Ds Index descriptors for each index the result
   * \tparam Operation The nullary operation taking n arguments
   * \return A matrix or array in which each component is the result of calling Operation with no arguments and which has
   * dimensions corresponding to Ds
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, std::size_t...indices, index_descriptor...Ds,
    detail::n_ary_operator<max_indices_of_v<PatternMatrix>>...Operations>
  requires
    ((fixed_index_descriptor<std::tuple_element_t<indices, std::tuple<Ds...>>>) and ...) and
    (sizeof...(Operations) == detail::count_index_dims<std::tuple<Ds...>, indices...>(std::index_sequence_for<Ds...> {}))
#else
  template<typename PatternMatrix, std::size_t...indices, typename...Ds, typename...Operations, std::enable_if_t<
    indexible<PatternMatrix> and (index_descriptor<Ds> and ...) and
    ((fixed_index_descriptor<std::tuple_element_t<indices, std::tuple<Ds...>>>) and ...) and
    (sizeof...(Operations) == detail::count_index_dims<std::tuple<Ds...>, indices...>(std::index_sequence_for<Ds...> {})) and
    (detail::n_ary_operator<Operations, max_indices_of_v<PatternMatrix>> and ...), int> = 0>
#endif
  constexpr auto
  n_ary_operation(std::index_sequence<indices...>, const std::tuple<Ds...>& d_tup, Operations&&...operations)
  {
    constexpr std::make_index_sequence<max_indices_of_v<PatternMatrix>> seq;
    using Scalar = std::common_type_t<std::decay_t<
      typename detail::n_ary_operator_traits<Operations, max_indices_of_v<PatternMatrix>>::type>...>;

    // One operation for all elements combined:
    if constexpr (sizeof...(Operations) == 1)
    {
      return detail::n_ary_operation_with_broadcasting_impl<PatternMatrix>(d_tup, std::forward<Operations>(operations)...);
    }
    // One operation for each element (only if index descriptors are fixed):
    else if constexpr (((dimension_size_of_v<Ds> != dynamic_size) and ...) and
      sizeof...(operations) == (dimension_size_of_v<Ds> * ...) and
      not (detail::is_invocable_with_indices<Operations&&>(seq) or ...))
    {
      return make_dense_writable_matrix_from<PatternMatrix, Scalar>(d_tup, std::forward<Operations>(operations)()...);
    }
    else
    {
      auto m = std::apply([](const auto&...ds){ return make_default_dense_writable_matrix_like<PatternMatrix, Scalar>(ds...); }, d_tup);
      auto operations_tuple = std::forward_as_tuple(std::forward<Operations>(operations)...);
      detail::nullary_iterate<0, sizeof...(Operations), indices...>(
        m, std::move(operations_tuple), d_tup,
        std::index_sequence<indices...> {},
        std::index_sequence_for<Ds...> {},
        std::index_sequence<> {},
        std::make_index_sequence<dimension_size_of_v<std::tuple_element_t<indices, std::tuple<Ds...>>>> {}...);
      return m;
    }
  }


  /**
   * \overload
   * \brief Perform a component-wise nullary operation, using a single operation for all elements.
   * \details Example:
   * - One operation for the entire matrix
   *   \code
   *     auto ds23 = std::tuple {Dimensions<2>{}, Dimensions<3>{}};
   *     auto M = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
   *     std::cout << n_ary_operation<M>(ds23, [](auto arg){return 7;}) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     7, 7, 7,
   *     7, 7, 7
   *   \endcode
   * \tparam PatternMatrix A matrix or array corresponding to the result type. Its dimensions need not match the specified dimensions Ds
   * \tparam Ds Index descriptors for each index the result
   * \tparam Operation The nullary operation
   * \return A matrix or array in which each component is the result of calling Operation with no arguments and which has
   * dimensions corresponding to Ds
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, index_descriptor...Ds, detail::n_ary_operator<max_indices_of_v<PatternMatrix>> Operation>
#else
  template<typename PatternMatrix, typename...Ds, typename Operation, std::enable_if_t<
    indexible<PatternMatrix> and (index_descriptor<Ds> and ...) and
    detail::n_ary_operator<Operation, max_indices_of_v<PatternMatrix>>, int> = 0>
#endif
  constexpr auto
  n_ary_operation(const std::tuple<Ds...>& d_tup, Operation&& operation)
  {
    return n_ary_operation<PatternMatrix>(std::index_sequence<>{}, d_tup, std::forward<Operation>(operation));
  }


  /**
   * \overload
   * \brief Perform a component-wise nullary operation, deriving the resulting size from a pattern matrix.
     * \details
     * - One operation for the entire matrix
     *   \code
     *     auto M = Eigen::Matrix<double, 2, 3>
     *     std::cout << n_ary_operation<M>(std::index_sequence<>{}, [](auto arg){return 7;}) << std::endl;
     *   \endcode
     *   Output:
     *   \code
     *     7, 7, 7,
     *     7, 7, 7
     *   \endcode
     * - One operation for each element
     *   \code
     *     std::cout << n_ary_operation<M>(std::index_sequence<0, 1>{}, []{return 4;}, []{return 5;}, []{return 6;}, []{return 7;}, []{return 8;}, []{return 9;});
     *   \endcode
     *   Output:
     *   \code
     *     4, 5, 6,
     *     7, 8, 9
     *   \endcode
     * - One operation for each row
     *   \code
     *     std::cout << n_ary_operation<M>(std::index_sequence<0>{}, []{return 5;}, []{return 6;});
     *   \endcode
     *   Output:
     *   \code
     *     5, 5, 5,
     *     6, 6, 6
     *   \endcode
     * - One operation for each column
     *   \code
     *     std::cout << n_ary_operation<M>(std::index_sequence<1>{}, []{return 5;}, []{return 6;}, []{return 7;});
     *   \endcode
     *   Output:
     *   \code
     *     5, 6, 7,
     *     5, 6, 7
     *   \endcode
   * \tparam PatternMatrix A matrix or array corresponding to the result type. Its dimensions need not match the specified dimensions Ds
   * \tparam indices The indices, if any, for which there is a distinct distribution for each slice based on that index.
   * \tparam Ds Index descriptors for each index the result
   * \tparam Operation The nullary operation
   * \return A matrix or array in which each component is the result of calling Operation with no arguments and which has
   * dimensions corresponding to Ds
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, std::size_t...indices, detail::n_ary_operator<max_indices_of_v<PatternMatrix>>...Operations>
  requires (not has_dynamic_dimensions<PatternMatrix>) and
    (sizeof...(Operations) == detail::pattern_index_dims<PatternMatrix, indices...>(
      std::make_index_sequence<max_indices_of_v<PatternMatrix>> {}))
#else
  template<typename PatternMatrix, std::size_t...indices, typename...Operations, std::enable_if_t<
    indexible<PatternMatrix> and (not has_dynamic_dimensions<PatternMatrix>) and
    (sizeof...(Operations) == detail::pattern_index_dims<PatternMatrix, indices...>(
      std::make_index_sequence<max_indices_of_v<PatternMatrix>> {})) and
    (detail::n_ary_operator<Operations, max_indices_of_v<PatternMatrix>> and ...), int> = 0>
#endif
  constexpr auto
  n_ary_operation(std::index_sequence<indices...> seq, Operations&&...operations)
  {
    auto d_tup = get_all_dimensions_of<PatternMatrix>();
    return n_ary_operation<PatternMatrix>(seq, d_tup, std::forward<Operations>(operations)...);
  }


  /**
   * \overload
   * \brief Perform a component-wise nullary operation on all elements, deriving the resulting size from a pattern matrix.
     * \details
     * - One operation for the entire matrix
     *   \code
     *     auto M = Eigen::Matrix<double, 2, 3>
     *     std::cout << n_ary_operation<M>([](auto arg){return 7;}) << std::endl;
     *   \endcode
     *   Output:
     *   \code
     *     7, 7, 7,
     *     7, 7, 7
     *   \endcode
     * - One operation for each element
     *   \code
     *     std::cout << n_ary_operation<M>([]{return 4;}, []{return 5;}, []{return 6;}, []{return 7;}, []{return 8;}, []{return 9;});
     *   \endcode
     *   Output:
     *   \code
     *     4, 5, 6,
     *     7, 8, 9
     *   \endcode
     * - One operation for each row
     *   \code
     *     std::cout << n_ary_operation<M>([]{return 5;}, []{return 6;});
     *   \endcode
     *   Output:
     *   \code
     *     5, 5, 5,
     *     6, 6, 6
     *   \endcode
     * - One operation for each column
     *   \code
     *     std::cout << n_ary_operation<M>([]{return 5;}, []{return 6;}, []{return 7;});
     *   \endcode
     *   Output:
     *   \code
     *     5, 6, 7,
     *     5, 6, 7
     *   \endcode
   * \tparam PatternMatrix A matrix or array corresponding to the result type. Its dimensions need not match the specified dimensions Ds
   * \tparam Operation The nullary operation
   * \return A matrix or array in which each component is the result of calling Operation with no arguments and which has
   * dimensions corresponding to Ds
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, detail::n_ary_operator<max_indices_of_v<PatternMatrix>> Operation>
  requires (not has_dynamic_dimensions<PatternMatrix>)
#else
  template<typename PatternMatrix, typename Operation, std::enable_if_t<
    indexible<PatternMatrix> and (not has_dynamic_dimensions<PatternMatrix>) and
    detail::n_ary_operator<Operation, max_indices_of_v<PatternMatrix>>, int> = 0>
#endif
  constexpr auto
  n_ary_operation(Operation&& operation)
  {
    return n_ary_operation<PatternMatrix>(std::index_sequence<>{}, std::forward<Operation>(operation));
  }


  // ----------- //
  //  randomize  //
  // ----------- //

  namespace detail
  {
#ifndef __cpp_concepts
    template<typename T, typename = void, typename = void>
    struct is_std_dist : std::false_type {};

    template<typename T>
    struct is_std_dist<T, std::void_t<typename T::result_type>, std::void_t<typename T::param_type>> : std::true_type {};
#endif


    template<typename random_number_generator>
    struct RandomizeGenerator
    {
      static auto& get()
      {
        static std::random_device rd;
        static std::decay_t<random_number_generator> gen {rd()};
        return gen;
      }
    };


    template<typename random_number_generator, typename distribution_type>
    struct RandomizeOp
    {
      template<typename G, typename D>
      RandomizeOp(G& g, D&& d) : generator{g}, distribution{std::forward<D>(d)} {}

      auto operator()() const
      {
        if constexpr (std::is_arithmetic_v<distribution_type>)
          return distribution;
        else
          return distribution(generator);
      }

    private:

      std::decay_t<random_number_generator>& generator;
      mutable std::decay_t<distribution_type> distribution;
    };


    template<typename G, typename D>
    RandomizeOp(G&, D&&) -> RandomizeOp<G, D>;

  } // namespace detail


  /**
   * \brief Create a matrix with random values selected from one or more random distributions.
   * \details This is essentially a specialized version of \ref n_ary_operation_with_indices with the unary operator
   * being a randomization function. The distributions are allocated to each element of the matrix, according to one
   * of the following options:
   *  - One distribution for all matrix elements. The following example constructs a 2-by-2 matrix (m) in which each
   *  element is a random value selected based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     using Mat = Eigen::Matrix<double, 2, 2>;
   *     auto g = std::mt19937 {};
   *     Mat m = randomize<Mat>(g, std::index_sequence<>{}, std::tuple {2, 2}, N {1.0, 0.3}));
   *   \endcode
   *  - One distribution for each matrix element. The following code constructs a 2-by-2 matrix m containing
   *  random values around mean 1.0, 2.0, 3.0, and 4.0 (in row-major order), with standard deviations of
   *  0.3, 0.3, 0.0 (by default, since no s.d. is specified as a parameter), and 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     auto m = randomize<Eigen::Matrix<double, 2, 2>>(g, std::index_sequence<0, 1>{},
   *       std::tuple {Dimensions<2>{}, Dimensions<2>{}}, N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3})));
   *   \endcode
   *  - One distribution for each row. The following code constructs a 3-by-2 (n) or 2-by-2 (o) matrices
   *  in which elements in each row are selected according to the three (n) or two (o) listed distribution
   *  parameters:
   *   \code
   *     auto n = randomize<Eigen::Matrix<double, 3, 2>>(g, std::index_sequence<0>{}, std::tuple {Dimensions<3>{}, 2},
   *       N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *     auto o = randomize<Eigen::Matrix<double, 2, 2>>(g, std::index_sequence<0>{}, std::tuple {Dimensions<2>{}, 2},
   *       N {1.0, 0.3}, N {2.0, 0.3})));
   *   \endcode
   *   Note that in the case of o, there is an ambiguity as to whether the listed distributions correspond to rows
   *   or columns. In case of such an ambiguity, this function assumes that the parameters correspond to the rows.
   *  - One distribution for each column. The following code constructs 2-by-3 matrix p
   *  in which elements in each column are selected according to the three listed distribution parameters:
   *   \code
   *     auto p = randomize<Eigen::Matrix<double, 2, 3>>(g, std::index_sequence<1>{}, std::tuple {2, Dimensions<3>{}},
   *       N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *   \endcode
   * \tparam PatternMatrix A matrix or array corresponding to the result type. Its dimensions need not match the
   * specified dimensions Ds
   * \tparam indices The indices, if any, for which there is a distinct distribution. If not provided, this can in some
   * cases be inferred from the number of Dists provided.
   * \tparam random_number_generator The random number generator (e.g., std::mt19937).
   * \tparam Ds Index descriptors for each index the result. They need not correspond to the dimensions of PatternMatrix.
   * \tparam Dists One or more distributions (e.g., std::normal_distribution<double>)
   * \sa n_ary_operation_with_indices
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, std::uniform_random_bit_generator random_number_generator,
    std::size_t...indices, index_descriptor...Ds, typename...Dists>
  requires (sizeof...(Ds) == max_indices_of_v<PatternMatrix>) and
    ((fixed_index_descriptor<std::tuple_element_t<indices, std::tuple<Ds...>>>) and ...) and
    (sizeof...(Dists) == detail::count_index_dims<std::tuple<Ds...>, indices...>(std::index_sequence_for<Ds...> {})) and
    ((std::is_arithmetic_v<std::decay_t<Dists>> or
      requires { typename std::decay_t<Dists>::result_type; typename std::decay_t<Dists>::param_type; }) and ...)
#else
  template<typename PatternMatrix, typename random_number_generator, std::size_t...indices, typename...Ds,
    typename...Dists, std::enable_if_t<indexible<PatternMatrix> and (index_descriptor<Ds> and ...) and
    (sizeof...(Ds) == max_indices_of_v<PatternMatrix>) and
    ((fixed_index_descriptor<std::tuple_element_t<indices, std::tuple<Ds...>>>) and ...) and
    (sizeof...(Dists) == detail::count_index_dims<std::tuple<Ds...>, indices...>(std::index_sequence_for<Ds...> {})) and
    ((std::is_arithmetic_v<std::decay_t<Dists>> or detail::is_std_dist<std::decay_t<Dists>>::value) and ...), int> = 0>
#endif
  constexpr auto
  randomize(random_number_generator& gen, std::index_sequence<indices...> seq, const std::tuple<Ds...>& ds_tuple, Dists&&...dists)
  {
    auto ret = n_ary_operation<PatternMatrix>(seq, ds_tuple,
      detail::RandomizeOp {gen, (std::forward<Dists>(dists))}...);

    if constexpr (sizeof...(Dists) == 1)
      return make_dense_writable_matrix_from(std::move(ret));
    else
      return ret;
  }


  /**
   * \overload
   * \brief Create a matrix with random values, using a default random number engine.
   * \details The distributions are allocated to each element of the matrix, according to one of the following options:
   *  - One distribution for all matrix elements. The following example constructs a 2-by-2 matrix (m) in which each
   *  element is a random value selected based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     using Mat = Eigen::Matrix<double, 2, 2>;
   *     Mat m = randomize<Mat>(std::index_sequence<>{}, std::tuple {2, 2}, N {1.0, 0.3}));
   *   \endcode
   *  - One distribution for each matrix element. The following code constructs a 2-by-2 matrix m containing
   *  random values around mean 1.0, 2.0, 3.0, and 4.0 (in row-major order), with standard deviations of
   *  0.3, 0.3, 0.0 (by default, since no s.d. is specified as a parameter), and 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     auto m = randomize<Eigen::Matrix<double, 2, 2>>(std::index_sequence<0, 1>{},
   *       std::tuple {Dimensions<2>{}, Dimensions<2>{}}, N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3})));
   *   \endcode
   *  - One distribution for each row. The following code constructs a 3-by-2 (n) or 2-by-2 (o) matrices
   *  in which elements in each row are selected according to the three (n) or two (o) listed distribution
   *  parameters:
   *   \code
   *     auto n = randomize<Eigen::Matrix<double, 3, 2>>(std::index_sequence<0>{}, std::tuple {Dimensions<3>{}, 2},
   *       N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *     auto o = randomize<Eigen::Matrix<double, 2, 2>>(std::index_sequence<0>{}, std::tuple {Dimensions<2>{}, 2},
   *       N {1.0, 0.3}, N {2.0, 0.3})));
   *   \endcode
   *   Note that in the case of o, there is an ambiguity as to whether the listed distributions correspond to rows
   *   or columns. In case of such an ambiguity, this function assumes that the parameters correspond to the rows.
   *  - One distribution for each column. The following code constructs 2-by-3 matrix p
   *  in which elements in each column are selected according to the three listed distribution parameters:
   *   \code
   *     auto p = randomize<Eigen::Matrix<double, 2, 3>>(std::index_sequence<1>{}, std::tuple {2, Dimensions<3>{}},
   *       N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *   \endcode
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, std::uniform_random_bit_generator random_number_generator = std::mt19937,
    std::size_t...indices, index_descriptor...Ds, typename...Dists>
  requires (sizeof...(Ds) == max_indices_of_v<PatternMatrix>) and
    ((fixed_index_descriptor<std::tuple_element_t<indices, std::tuple<Ds...>>>) and ...) and
    (sizeof...(Dists) == detail::count_index_dims<std::tuple<Ds...>, indices...>(std::index_sequence_for<Ds...> {})) and
    std::constructible_from<random_number_generator, typename std::random_device::result_type> and
    ((std::is_arithmetic_v<std::decay_t<Dists>> or
      requires { typename std::decay_t<Dists>::result_type; typename std::decay_t<Dists>::param_type; }) and ...)
#else
  template<typename PatternMatrix, typename random_number_generator = std::mt19937,
      std::size_t...indices, typename...Ds, typename...Dists,
    std::enable_if_t<indexible<PatternMatrix> and (index_descriptor<Ds> and ...) and
    (sizeof...(Ds) == max_indices_of_v<PatternMatrix>) and
    ((fixed_index_descriptor<std::tuple_element_t<indices, std::tuple<Ds...>>>) and ...) and
    (sizeof...(Dists) == detail::count_index_dims<std::tuple<Ds...>, indices...>(std::index_sequence_for<Ds...> {})) and
    std::is_constructible_v<random_number_generator, typename std::random_device::result_type> and
    ((std::is_arithmetic_v<std::decay_t<Dists>> or detail::is_std_dist<std::decay_t<Dists>>::value) and ...), int> = 0>
#endif
  constexpr auto
  randomize(std::index_sequence<indices...> seq, const std::tuple<Ds...>& d_tuple, Dists&&...dists)
  {
    auto& gen = detail::RandomizeGenerator<random_number_generator>::get();
    return randomize<PatternMatrix>(gen, seq, d_tuple, std::forward<Dists>(dists)...);
  }


  /**
   * \overload
   * \brief Create a matrix with random values, using a default random number engine.
   * \details The distributions are allocated to each element of the matrix, according to one of the following options:
   *  - One distribution for all matrix elements. The following example constructs a 2-by-2 matrix (m) in which each
   *  element is a random value selected based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     using Mat = Eigen::Matrix<double, 2, 2>;
   *     Mat m = randomize<Mat>(std::tuple {2, 2}, N {1.0, 0.3}));
   *   \endcode
   *  - One distribution for each matrix element. The following code constructs a 2-by-2 matrix m containing
   *  random values around mean 1.0, 2.0, 3.0, and 4.0 (in row-major order), with standard deviations of
   *  0.3, 0.3, 0.0 (by default, since no s.d. is specified as a parameter), and 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     auto m = randomize<Eigen::Matrix<double, 2, 2>>(std::tuple {Dimensions<2>{}, Dimensions<2>{}},
   *       N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3})));
   *   \endcode
   *  - One distribution for each row. The following code constructs a 3-by-2 (n) or 2-by-2 (o) matrices
   *  in which elements in each row are selected according to the three (n) or two (o) listed distribution
   *  parameters:
   *   \code
   *     auto n = randomize<Eigen::Matrix<double, 3, 2>>(std::tuple {Dimensions<3>{}, 2}, N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *     auto o = randomize<Eigen::Matrix<double, 2, 2>>(std::tuple {Dimensions<2>{}, 2}, N {1.0, 0.3}, N {2.0, 0.3})));
   *   \endcode
   *   Note that in the case of o, there is an ambiguity as to whether the listed distributions correspond to rows
   *   or columns. In case of such an ambiguity, this function assumes that the parameters correspond to the rows.
   *  - One distribution for each column. The following code constructs 2-by-3 matrix p
   *  in which elements in each column are selected according to the three listed distribution parameters:
   *   \code
   *     auto p = randomize<Eigen::Matrix<double, 2, 3>>(std::tuple {2, Dimensions<3>{}}, N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *   \endcode
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, std::uniform_random_bit_generator random_number_generator = std::mt19937,
    index_descriptor...Ds, typename Dist>
  requires (sizeof...(Ds) == max_indices_of_v<PatternMatrix>) and
    std::constructible_from<random_number_generator, typename std::random_device::result_type> and
    (std::is_arithmetic_v<std::decay_t<Dist>> or
      requires { typename std::decay_t<Dist>::result_type; typename std::decay_t<Dist>::param_type; })
#else
  template<typename PatternMatrix, typename random_number_generator = std::mt19937, typename...Ds, typename Dist,
    std::enable_if_t<indexible<PatternMatrix> and (index_descriptor<Ds> and ...) and
    (sizeof...(Ds) == max_indices_of_v<PatternMatrix>) and
    std::is_constructible_v<random_number_generator, typename std::random_device::result_type> and
    (std::is_arithmetic_v<std::decay_t<Dist>> or detail::is_std_dist<std::decay_t<Dist>>::value), int> = 0>
#endif
  constexpr auto
  randomize(const std::tuple<Ds...>& d_tuple, Dist&& dist)
  {
    return randomize<PatternMatrix, random_number_generator>(std::index_sequence<>{}, d_tuple, std::forward<Dist>(dist));
  }


  /**
   * \overload
   * \brief Fill a fixed-sized matrix with random values selected from one or more random distributions.
   * \details The distributions are allocated to each element of the matrix, according to one of the following options:
   *  - One distribution for all matrix elements. The following example constructs a 2-by-2 matrix (m) in which each
   *  element is a random value selected based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     using Mat = Eigen::Matrix<double, 2, 2>;
   *     Mat m = randomize<Mat>(std::index_sequence<>, N {1.0, 0.3}));
   *   \endcode
   *  - One distribution for each matrix element. The following code constructs a 2-by-2 matrix m containing
   *  random values around mean 1.0, 2.0, 3.0, and 4.0 (in row-major order), with standard deviations of
   *  0.3, 0.3, 0.0 (by default, since no s.d. is specified as a parameter), and 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     auto m = randomize<Eigen::Matrix<double, 2, 2>>(std::index_sequence<0, 1>, N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3})));
   *   \endcode
   *  - One distribution for each row. The following code constructs a 3-by-2 (n) or 2-by-2 (o) matrices
   *  in which elements in each row are selected according to the three (n) or two (o) listed distribution
   *  parameters:
   *   \code
   *     auto n = randomize<Eigen::Matrix<double, 3, 2>>(std::index_sequence<0>, N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *     auto o = randomize<Eigen::Matrix<double, 2, 2>>(std::index_sequence<0>, N {1.0, 0.3}, N {2.0, 0.3})));
   *   \endcode
   *   Note that in the case of o, there is an ambiguity as to whether the listed distributions correspond to rows
   *   or columns. In case of such an ambiguity, this function assumes that the parameters correspond to the rows.
   *  - One distribution for each column. The following code constructs 2-by-3 matrix p
   *  in which elements in each column are selected according to the three listed distribution parameters:
   *   \code
   *     auto p = randomize<Eigen::Matrix<double, 2, 3>>(std::index_sequence<1>, N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *   \endcode
   * \tparam PatternMatrix A fixed-size matrix
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, std::uniform_random_bit_generator random_number_generator = std::mt19937,
    std::size_t...indices, typename...Dists>
  requires
    (not has_dynamic_dimensions<PatternMatrix>) and
    (sizeof...(Dists) == detail::pattern_index_dims<PatternMatrix, indices...>(
      std::make_index_sequence<max_indices_of_v<PatternMatrix>> {})) and
    std::constructible_from<random_number_generator, typename std::random_device::result_type> and
    ((std::is_arithmetic_v<Dists> or requires { typename Dists::result_type; typename Dists::param_type; }) and ...)
#else
  template<typename PatternMatrix, typename random_number_generator = std::mt19937, std::size_t...indices,
    typename...Dists, std::enable_if_t<indexible<PatternMatrix> and (not has_dynamic_dimensions<PatternMatrix>) and
    (sizeof...(Dists) == detail::pattern_index_dims<PatternMatrix, indices...>(
      std::make_index_sequence<max_indices_of_v<PatternMatrix>> {})) and
    std::is_constructible_v<random_number_generator, typename std::random_device::result_type> and
    ((std::is_arithmetic_v<Dists> or detail::is_std_dist<Dists>::value) and ...), int> = 0>
#endif
  constexpr auto
  randomize(std::index_sequence<indices...> seq, Dists&&...dists)
  {
    auto d_tup = get_all_dimensions_of<PatternMatrix>();
    return randomize<PatternMatrix, random_number_generator>(seq, d_tup, std::forward<Dists>(dists)...);
  }


  /**
   * \overload
   * \brief Fill a fixed-sized matrix with random values selected from a random distribution.
   * \details The distributions are allocated to each element of the matrix, according to one of the following options:
   *  - One distribution for all matrix elements. The following example constructs a 2-by-2 matrix (m) in which each
   *  element is a random value selected based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     using Mat = Eigen::Matrix<double, 2, 2>;
   *     Mat m = randomize<Mat>(N {1.0, 0.3}));
   *   \endcode
   * \tparam PatternMatrix A fixed-size matrix
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, std::uniform_random_bit_generator random_number_generator = std::mt19937, typename Dist>
  requires
    (not has_dynamic_dimensions<PatternMatrix>) and
    std::constructible_from<random_number_generator, typename std::random_device::result_type> and
    (std::is_arithmetic_v<Dist> or requires { typename Dist::result_type; typename Dist::param_type; })
#else
  template<typename PatternMatrix, typename random_number_generator = std::mt19937, typename Dist,
    std::enable_if_t<indexible<PatternMatrix> and (not has_dynamic_dimensions<PatternMatrix>) and
    std::is_constructible_v<random_number_generator, typename std::random_device::result_type> and
    (std::is_arithmetic_v<Dist> or detail::is_std_dist<Dist>::value), int> = 0>
#endif
  constexpr auto
  randomize(Dist&& dist)
  {
    return randomize<PatternMatrix, random_number_generator>(std::index_sequence<>{}, std::forward<Dist>(dist));
  }


  // -------- //
  //  reduce  //
  // -------- //

  namespace detail
  {
    template<std::size_t I, std::size_t...index, typename DTup>
    constexpr decltype(auto) get_reduced_index(DTup&& d_tup)
    {
      if constexpr (((I == index) or ...))
      {
        using T = std::tuple_element_t<I, DTup>;
        if constexpr (has_uniform_dimension_type<T>) return uniform_dimension_type_of_t<T>{};
        else return Dimensions<1>{};
      }
      else return std::get<I>(std::forward<DTup>(d_tup));
    }


    template<std::size_t...index, typename T, std::size_t...I>
    constexpr auto make_zero_matrix_reduction(T&& t, std::index_sequence<I...>)
    {
      decltype(auto) d_tup = get_all_dimensions_of(std::forward<T>(t));
      return make_zero_matrix_like<T>(get_reduced_index<I, index...>(std::forward<decltype(d_tup)>(d_tup))...);
    }


    template<auto constant, std::size_t...index, typename T, std::size_t...I>
    constexpr auto make_constant_matrix_reduction(T&& t, std::index_sequence<I...>)
    {
      decltype(auto) d_tup = get_all_dimensions_of(std::forward<T>(t));
      return make_constant_matrix_like<T, constant>(get_reduced_index<I, index...>(std::forward<decltype(d_tup)>(d_tup))...);
    }


    template<auto constant, std::size_t index, typename T, std::size_t...I>
    constexpr auto make_constant_diagonal_matrix_reduction(T&& t, std::index_sequence<I...>)
    {
      // \todo Handle 3+ dimensional constant diagonal tensors
      static_assert(index == 0 or index == 1);
      decltype(auto) d_tup = get_all_dimensions_of(std::forward<T>(t));
      return make_constant_matrix_like<T, constant>(get_reduced_index<I, index>(std::forward<decltype(d_tup)>(d_tup))...);
    }


    template<typename T, std::size_t...indices, std::size_t...I>
    constexpr std::size_t count_reduced_dimensions(std::index_sequence<I...>)
    {
      return ([]{
        if constexpr (is_index_match<I, indices...>()) return index_dimension_of_v<T, I>;
        else return 1;
      }() * ...);
    }


    template<std::size_t...indices, typename T, std::size_t...I>
    constexpr std::size_t count_reduced_dimensions(const T& t, std::index_sequence<I...>)
    {
      return ([](const T& t){
        if constexpr (is_index_match<I, indices...>()) return get_index_dimension_of<I>(t);
        else return 1;
      }(t) * ...);
    }


    template<std::size_t dim, typename BinaryFunction, std::size_t...index, typename Scalar>
    constexpr Scalar calc_reduce_constant(Scalar constant)
    {
      if constexpr (dim <= 1)
        return constant;
      else if constexpr (is_plus<BinaryFunction>::value)
        return constant * dim;
      else if constexpr (is_multiplies<BinaryFunction>::value)
        return OpenKalman::internal::constexpr_pow(constant, dim);
      else
        return BinaryFunction{}(constant, calc_reduce_constant<dim - 1, BinaryFunction>(constant));
    }


    template<std::size_t...index, typename BinaryFunction, typename Scalar>
    constexpr Scalar calc_reduce_constant(std::size_t dim, const BinaryFunction& b, Scalar constant)
    {
      if (dim <= 1)
        return constant;
      else if constexpr (is_plus<BinaryFunction>::value)
        return constant * dim;
      else if constexpr (is_multiplies<BinaryFunction>::value)
        return std::pow(constant, dim);
      else
        return b(constant, calc_reduce_constant(dim - 1, b, constant));
    }


    template<typename Arg, std::size_t...I>
    constexpr bool has_uniform_reduction_indices(std::index_sequence<I...>)
    {
      return ((has_uniform_dimension_type<coefficient_types_of_t<Arg, I>> or dynamic_dimension<Arg, I>) and ...);
    }


    template<typename BinaryFunction, typename Arg, std::size_t...indices>
    constexpr scalar_type_of_t<Arg>
    reduce_all_indices(const BinaryFunction& b, Arg&& arg, std::index_sequence<indices...>)
    {
      if constexpr (zero_matrix<Arg> and (is_plus<BinaryFunction>::value or is_multiplies<BinaryFunction>::value))
      {
        return 0;
      }
      else if constexpr (constant_matrix<Arg>)
      {
        using Scalar = scalar_type_of_t<Arg>;
        constexpr Scalar c = constant_coefficient_v<Arg>;
        constexpr auto seq = std::make_index_sequence<max_indices_of_v<Arg>> {};
        constexpr bool fixed_reduction_dims = not (dynamic_dimension<Arg, indices> or ...);

        if constexpr (fixed_reduction_dims and is_constexpr_n_ary_function<BinaryFunction, Scalar, Scalar>::value)
        {
          constexpr std::size_t dim = count_reduced_dimensions<Arg, indices...>(seq);
          return calc_reduce_constant<dim, BinaryFunction, indices...>(c);
        }
        else
        {
          std::size_t dim = count_reduced_dimensions<indices...>(arg, seq);
          return calc_reduce_constant<indices...>(dim, b, c);
        }
      }
      else
      {
        decltype(auto) red = interface::ArrayOperations<std::decay_t<Arg>>::template reduce<indices...>(b, std::forward<Arg>(arg));
        using Red = decltype(red);

        static_assert(scalar_type<Red> or
          ((index_dimension_of_v<Red, indices> == 1 or index_dimension_of_v<Red, indices> == dynamic_size) and ...));

        if constexpr (scalar_type<Red>)
          return std::forward<Red>(red);
        else if constexpr (element_gettable<Red, decltype(indices)...>)
          return get_element(std::forward<Red>(red), static_cast<decltype(indices)>(0)...);
        else
          return interface::LinearAlgebra<std::decay_t<Red>>::trace(std::forward<Red>(red));
      }
    }

  } // namespace detail


  /**
   * \brief Perform a complete reduction based on an associative binary function, and return a scalar.
   * \details The binary function must be associative. (This is not enforced, but the order of operation is undefined.)
   * \tparam BinaryFunction A binary function invocable with two values of type <code>scalar_type_of_t<Arg></code>.
   * It must be an associative function. Preferably, it should be a constexpr function, and even more preferably,
   * it should be a standard c++ function such as std::plus or std::multiplies.
   * \tparam Arg The tensor
   * \returns A scalar representing a complete reduction.
   */
#ifdef __cpp_concepts
  template<typename BinaryFunction, indexible Arg> requires
    std::is_invocable_r_v<scalar_type_of_t<Arg>, BinaryFunction&&, scalar_type_of_t<Arg>, scalar_type_of_t<Arg>> and
    (detail::has_uniform_reduction_indices<Arg>(std::make_index_sequence<max_indices_of_v<Arg>> {}))
#else
  template<typename BinaryFunction, typename Arg, std::enable_if_t<indexible<Arg> and
    std::is_invocable_r<typename scalar_type_of<Arg>::type, BinaryFunction&&,
      typename scalar_type_of<Arg>::type, typename scalar_type_of<Arg>::type>::value and
    (detail::has_uniform_reduction_indices<Arg>(std::make_index_sequence<max_indices_of_v<Arg>> {})), int> = 0>
#endif
  constexpr decltype(auto)
  reduce(const BinaryFunction& b, Arg&& arg)
  {
    constexpr auto max_indices = max_indices_of_v<Arg>;
    return detail::reduce_all_indices(b, std::forward<Arg>(arg), std::make_index_sequence<max_indices> {});
  }


  /**
   * \overload
   * \brief Perform a partial reduction based on an associative binary function, across one or more indices.
   * \details The binary function must be associative. (This is not enforced, but the order of operation is undefined.)
   * \tparam index an index to be reduced. For example, if the index is 0, the result will have only one row.
   * If the index is 1, the result will have only one column.
   * \tparam indices Other indicesto be reduced. Because the binary function is associative, the order
   * of the indices does not matter.
   * \tparam BinaryFunction A binary function invocable with two values of type <code>scalar_type_of_t<Arg></code>.
   * It must be an associative function. Preferably, it should be a constexpr function, and even more preferably,
   * it should be a standard c++ function such as std::plus or std::multiplies.
   * \tparam Arg The tensor
   * \returns A vector or tensor with reduced dimensions.
   */
#ifdef __cpp_concepts
  template<std::size_t index, std::size_t...indices, typename BinaryFunction, indexible Arg> requires
    ((index < max_indices_of_v<Arg>) and ... and (indices < max_indices_of_v<Arg>)) and
    std::is_invocable_r_v<scalar_type_of_t<Arg>, BinaryFunction&&, scalar_type_of_t<Arg>, scalar_type_of_t<Arg>> and
    (detail::has_uniform_reduction_indices<Arg>(std::index_sequence<index, indices...> {}))
#else
  template<std::size_t index, std::size_t...indices, typename BinaryFunction, typename Arg, std::enable_if_t<
    indexible<Arg> and ((index < max_indices_of<Arg>::value) and ... and (indices < max_indices_of<Arg>::value)) and
    std::is_invocable_r<typename scalar_type_of<Arg>::type, BinaryFunction&&,
      typename scalar_type_of<Arg>::type, typename scalar_type_of<Arg>::type>::value and
    (detail::has_uniform_reduction_indices<Arg>(std::index_sequence<index, indices...> {})), int> = 0>
#endif
  constexpr auto
  reduce(const BinaryFunction& b, Arg&& arg)
  {
    constexpr auto max_indices = max_indices_of_v<Arg>;
    constexpr std::make_index_sequence<max_indices> seq;

    if constexpr (covariance<Arg>)
    {
      using RC = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      using CC = std::conditional_t<((index == 1) or ... or (indices == 1)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 1>>, coefficient_types_of_t<Arg, 1>>;
      auto m = reduce<index, indices...>(b, to_covariance_nestable(std::forward<Arg>(arg)));
      return Matrix<RC, CC, decltype(m)> {std::move(m)};
    }
    else if constexpr(mean<Arg> and ((index != 0) or ... or (indices != 0)))
    {
      using C = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      auto m = from_euclidean<C>(reduce<index, indices...>(b, nested_matrix(to_euclidean(std::forward<Arg>(arg)))));
      return Mean<C, decltype(m)> {std::move(m)};
    }
    else if constexpr (euclidean_transformed<Arg> and ((index != 0) or ... or (indices != 0)))
    {
      using C = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      auto m = reduce<index, indices...>(b, nested_matrix(std::forward<Arg>(arg)));
      return EuclideanMean<C, decltype(m)> {std::move(m)};
    }
    else if constexpr (typed_matrix<Arg>)
    {
      using RC = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      using CC = std::conditional_t<((index == 1) or ... or (indices == 1)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 1>>, coefficient_types_of_t<Arg, 1>>;
      auto m = reduce<index, indices...>(b, nested_matrix(std::forward<Arg>(arg)));
      return Matrix<RC, CC, decltype(m)> {std::move(m)};
    }
    else if constexpr (index_dimension_of_v<Arg, index> == 1)
    {
      if constexpr (sizeof...(indices) == 0) return std::forward<Arg>(arg);
      else return reduce<indices...>(b, std::forward<Arg>(arg));
    }
    else if constexpr (zero_matrix<Arg> and (detail::is_plus<BinaryFunction>::value or detail::is_multiplies<BinaryFunction>::value))
    {
      return detail::make_zero_matrix_reduction<index, indices...>(std::forward<Arg>(arg), seq);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      using Scalar = scalar_type_of_t<Arg>;
      constexpr Scalar c_arg = constant_coefficient_v<Arg>;
      constexpr bool fixed_reduction_dims = not (dynamic_dimension<Arg, index> or ... or dynamic_dimension<Arg, indices>);

      if constexpr (fixed_reduction_dims and detail::is_constexpr_n_ary_function<BinaryFunction, Scalar, Scalar>::value)
      {
        constexpr std::size_t dim = detail::count_reduced_dimensions<Arg, index, indices...>(seq);
        constexpr auto c = detail::calc_reduce_constant<dim, BinaryFunction, index, indices...>(c_arg);
# if __cpp_nontype_template_args >= 201911L
        return detail::make_constant_matrix_reduction<c, indices...>(std::forward<Arg>(arg), seq);
# else
        constexpr auto c_integral = static_cast<std::intmax_t>(c);
        if constexpr (are_within_tolerance(c, static_cast<Scalar>(c_integral)))
        {
          return detail::make_constant_matrix_reduction<c_integral, index, indices...>(std::forward<Arg>(arg), seq);
        }
        else
        {
          auto red = detail::make_constant_matrix_reduction<1, index, indices...>(std::forward<Arg>(arg), seq);
          return make_self_contained(c * to_native_matrix<Arg>(std::move(red)));
        }
# endif
      }
      else
      {
        std::size_t dim = detail::count_reduced_dimensions<index, indices...>(arg, seq);
        auto c = detail::calc_reduce_constant<index, indices...>(dim, b, c_arg);
        auto red = detail::make_constant_matrix_reduction<1, index, indices...>(std::forward<Arg>(arg), seq);
        return make_self_contained(c * to_native_matrix<Arg>(std::move(red)));
      }
    }
    //else if constexpr (constant_diagonal_matrix<Arg>)
    //{
    //  return const_diagonal_reduce<indices...>(b, std::forward<Arg>(arg), seq);
    //}
    else
    {
      return interface::ArrayOperations<std::decay_t<Arg>>::template reduce<index, indices...>(b, std::forward<Arg>(arg));
    }
  }


  // ---------------- //
  //  average_reduce  //
  // ---------------- //

  /**
   * \brief Perform a complete reduction by taking the average along all indices and returning a scalar value.
   * \returns A scalar representing the average of all components.
   */
#ifdef __cpp_concepts
  template<indexible Arg> requires
    (detail::has_uniform_reduction_indices<Arg>(std::make_index_sequence<max_indices_of_v<Arg>> {}))
#else
  template<typename Arg, std::enable_if_t<indexible<Arg> and
    detail::has_uniform_reduction_indices<Arg>(std::make_index_sequence<max_indices_of_v<Arg>> {}), int> = 0>
#endif
  constexpr decltype(auto)
  average_reduce(Arg&& arg) noexcept
  {
    if constexpr (zero_matrix<Arg>)
      return 0;
    else if constexpr (constant_matrix<Arg>)
      return constant_coefficient_v<Arg>;
    else
      return reduce(std::plus<scalar_type_of_t<Arg>> {}, std::forward<Arg>(arg)) / (
        std::apply([](const auto&...d){ return (get_dimension_size_of(d) * ...); }, get_all_dimensions_of(arg)));
  }


  /**
   * \overload
   * \brief Perform a partial reduction by taking the average along one or more indices.
   * \tparam index an index to be reduced. For example, if the index is 0, the result will have only one row.
   * If the index is 1, the result will have only one column.
   * \tparam indices Other indicesto be reduced. Because the binary function is associative, the order
   * of the indices does not matter.
   * \returns A vector or tensor with reduced dimensions.
   */
#ifdef __cpp_concepts
  template<std::size_t index, std::size_t...indices, indexible Arg> requires
    ((index < max_indices_of_v<Arg>) and ... and (indices < max_indices_of_v<Arg>)) and
    (detail::has_uniform_reduction_indices<Arg>(std::index_sequence<index, indices...> {}))
#else
  template<std::size_t index, std::size_t...indices, typename Arg, std::enable_if_t<indexible<Arg> and
    ((index < max_indices_of_v<Arg>) and ... and (indices < max_indices_of_v<Arg>)) and
    (detail::has_uniform_reduction_indices<Arg>(std::index_sequence<index, indices...> {})), int> = 0>
#endif
  constexpr auto
  average_reduce(Arg&& arg) noexcept
  {
    using Scalar = scalar_type_of_t<Arg>;
    constexpr auto max_indices = max_indices_of_v<Arg>;
    constexpr std::make_index_sequence<max_indices> seq;

    if constexpr (covariance<Arg>)
    {
      using RC = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      using CC = std::conditional_t<((index == 1) or ... or (indices == 1)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 1>>, coefficient_types_of_t<Arg, 1>>;
      auto m = average_reduce<index, indices...>(to_covariance_nestable(std::forward<Arg>(arg)));
      return Matrix<RC, CC, decltype(m)> {std::move(m)};
    }
    else if constexpr(mean<Arg> and ((index != 0) or ... or (indices != 0)))
    {
      using C = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      auto m = from_euclidean<C>(average_reduce<index, indices...>(nested_matrix(to_euclidean(std::forward<Arg>(arg)))));
      return Mean<C, decltype(m)> {std::move(m)};
    }
    else if constexpr (euclidean_transformed<Arg> and ((index != 0) or ... or (indices != 0)))
    {
      using C = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      auto m = average_reduce<index, indices...>(nested_matrix(std::forward<Arg>(arg)));
      return EuclideanMean<C, decltype(m)> {std::move(m)};
    }
    else if constexpr (typed_matrix<Arg>)
    {
      using RC = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      using CC = std::conditional_t<((index == 1) or ... or (indices == 1)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 1>>, coefficient_types_of_t<Arg, 1>>;
      auto m = average_reduce<index, indices...>(nested_matrix(std::forward<Arg>(arg)));
      return Matrix<RC, CC, decltype(m)> {std::move(m)};
    }
    else if constexpr (index_dimension_of_v<Arg, index> == 1)
    {
      if constexpr (sizeof...(indices) == 0) return std::forward<Arg>(arg);
      else return average_reduce<indices...>(std::forward<Arg>(arg));
    }
    else if constexpr (zero_matrix<Arg>)
    {
      return detail::make_zero_matrix_reduction<index, indices...>(std::forward<Arg>(arg), seq);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      constexpr Scalar c = constant_coefficient_v<Arg>;
# if __cpp_nontype_template_args >= 201911L
      return detail::make_constant_matrix_reduction<c, index, indices...>(std::forward<Arg>(arg), seq);
# else
      constexpr auto c_integral = static_cast<std::intmax_t>(c);
      if constexpr (are_within_tolerance(c, static_cast<Scalar>(c_integral)))
      {
        return detail::make_constant_matrix_reduction<c_integral, index, indices...>(std::forward<Arg>(arg), seq);
      }
      else
      {
        auto red = detail::make_constant_matrix_reduction<1, index, indices...>(std::forward<Arg>(arg), seq);
        return make_self_contained(c * to_native_matrix<Arg>(red));
      }
# endif
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      if constexpr (not dynamic_dimension<Arg, 0>)
      {
        constexpr auto c = static_cast<Scalar>(constant_diagonal_coefficient_v<Arg>) / index_dimension_of_v<Arg, 0>;
# if __cpp_nontype_template_args >= 201911L
        return average_reduce<indices...>(detail::make_constant_diagonal_matrix_reduction<c, index>(std::forward<Arg>(arg), seq));
# else
        constexpr auto c_integral = static_cast<std::intmax_t>(c);
        if constexpr (are_within_tolerance(c, static_cast<Scalar>(c_integral)))
        {
          auto ret = detail::make_constant_diagonal_matrix_reduction<c_integral, index>(std::forward<Arg>(arg), seq);
          if constexpr (sizeof...(indices) == 0) return ret;
          else return average_reduce<indices...>(std::move(ret));
        }
        else
        {
          auto ret = detail::make_constant_diagonal_matrix_reduction<1, index>(std::forward<Arg>(arg), seq);
          if constexpr (sizeof...(indices) == 0) return make_self_contained(c * std::move(ret));
          else return make_self_contained(c * to_native_matrix<Arg>(average_reduce<indices...>(std::move(ret))));
        }
# endif
      }
      else
      {
        auto c = static_cast<Scalar>(constant_diagonal_coefficient_v<Arg>) / get_index_dimension_of<0>(arg);
        auto ret = detail::make_constant_diagonal_matrix_reduction<1, index>(std::forward<Arg>(arg), seq);
        if constexpr (sizeof...(indices) == 0) return make_self_contained(c * std::move(ret));
        else return make_self_contained(c * average_reduce<indices...>(std::move(ret)));
      }
    }
    else
    {
      return make_self_contained(reduce<index, indices...>(std::plus<Scalar> {}, std::forward<Arg>(arg)) /
        (get_index_dimension_of<index>(arg) * ... * get_index_dimension_of<indices>(arg)));
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
      else if constexpr (not has_dynamic_dimensions<Arg> and row_dimension_of_v<Arg> == column_dimension_of_v<Arg>)
        return conjugate(std::forward<Arg>(arg));
      else
        return interface::LinearAlgebra<std::decay_t<Arg>>::adjoint(std::forward<Arg>(arg));
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      if constexpr (std::imag(constant_diagonal_coefficient_v<Arg>) == 0)
        return transpose(std::forward<Arg>(arg));
      else if constexpr (not has_dynamic_dimensions<Arg> and row_dimension_of_v<Arg> == column_dimension_of_v<Arg>)
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
  template<typename Arg> requires has_dynamic_dimensions<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<has_dynamic_dimensions<Arg> or square_matrix<Arg>, int> = 0>
#endif
  constexpr auto determinant(Arg&& arg)
  {
    if constexpr (has_dynamic_dimensions<Arg>) if (get_dimensions_of<0>(arg) != get_dimensions_of<1>(arg))
      throw std::domain_error {
        "In determinant, rows of arg (" + std::to_string(get_index_dimension_of<0>(arg)) + ") do not match columns of arg (" +
        std::to_string(get_index_dimension_of<1>(arg)) + ")"};

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
        return std::pow(constant_diagonal_coefficient_v<Arg>, get_index_dimension_of<0>(arg));
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
  template<typename Arg> requires has_dynamic_dimensions<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<(has_dynamic_dimensions<Arg> or square_matrix<Arg>), int> = 0>
#endif
  constexpr auto trace(Arg&& arg)
  {
    if constexpr (has_dynamic_dimensions<Arg>) if (get_dimensions_of<0>(arg) != get_dimensions_of<1>(arg))
      throw std::domain_error {
        "In trace, rows of arg (" + std::to_string(get_index_dimension_of<0>(arg)) + ") do not match columns of arg (" +
        std::to_string(get_index_dimension_of<1>(arg)) + ")"};

    using Scalar = scalar_type_of_t<Arg>;

    if constexpr (identity_matrix<Arg>)
    {
      return Scalar(get_index_dimension_of<0>(arg));
    }
    else if constexpr (zero_matrix<Arg>)
    {
      return Scalar(0);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      return Scalar(constant_coefficient_v<Arg> * get_index_dimension_of<0>(arg));
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      return Scalar(constant_diagonal_coefficient_v<Arg> * get_index_dimension_of<0>(arg));
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
    (has_dynamic_dimensions<A> or square_matrix<A>)
#else
  template<TriangleType t, typename A, typename U, typename Alpha, std::enable_if_t<
    std::is_convertible_v<Alpha, const scalar_type_of_t<A>> and
    (dynamic_rows<U> or dynamic_rows<A> or
      row_dimension_of<std::decay_t<U>>::value == row_dimension_of<std::decay_t<A>>::value) and
    (has_dynamic_dimensions<A> or square_matrix<A>), int> = 0>
#endif
  inline decltype(auto)
  rank_update_self_adjoint(A&& a, U&& u, Alpha alpha = 1)
  {
    if constexpr (dynamic_rows<A> or dynamic_rows<U>) if (get_index_dimension_of<0>(a) != get_index_dimension_of<0>(u))
      throw std::domain_error {
        "In rank_update_self_adjoint, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
        ") do not match rows of u (" + std::to_string(get_index_dimension_of<0>(u)) + ")"};

    if constexpr (has_dynamic_dimensions<A>) if (get_index_dimension_of<0>(a) != get_index_dimension_of<1>(a))
      throw std::domain_error {
        "In rank_update_self_adjoint, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
        ") do not match columns of a (" + std::to_string(get_index_dimension_of<1>(a)) + ")"};

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
    (has_dynamic_dimensions<A> or square_matrix<A>)
#else
  template<typename A, typename U, typename Alpha, std::enable_if_t<
    std::is_convertible_v<Alpha, const scalar_type_of_t<A>> and
    (dynamic_rows<U> or dynamic_rows<A> or row_dimension_of<U>::value == row_dimension_of<A>::value) and
    (has_dynamic_dimensions<A> or square_matrix<A>), int> = 0>
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
    (has_dynamic_dimensions<A> or square_matrix<A>)
# else
  template<TriangleType t, typename A, typename U, typename Alpha, std::enable_if_t<
    std::is_convertible_v<Alpha, const scalar_type_of_t<A>> and
    (t != TriangleType::lower or lower_triangular_matrix<A> or not upper_triangular_matrix<A>) and
    (t != TriangleType::upper or upper_triangular_matrix<A> or not lower_triangular_matrix<A>) and
    (dynamic_rows<A> or dynamic_rows<U> or row_dimension_of<A>::value == row_dimension_of<U>::value) and
    (has_dynamic_dimensions<A> or square_matrix<A>), int> = 0>
# endif
  inline decltype(auto)
  rank_update_triangular(A&& a, U&& u, Alpha alpha = 1)
  {
    if constexpr (dynamic_rows<A> or dynamic_rows<U>) if (get_index_dimension_of<0>(a) != get_index_dimension_of<0>(u))
      throw std::domain_error {
        "In rank_update_triangular, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
        ") do not match rows of u (" + std::to_string(get_index_dimension_of<0>(u)) + ")"};

    if constexpr (has_dynamic_dimensions<A>) if (get_index_dimension_of<0>(a) != get_index_dimension_of<1>(a))
      throw std::domain_error {
        "In rank_update_triangular, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
        ") do not match columns of a (" + std::to_string(get_index_dimension_of<1>(a)) + ")"};

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
    (has_dynamic_dimensions<A> or square_matrix<A>)
# else
  template<typename A, typename U, typename Alpha, std::enable_if_t<
    std::is_convertible_v<Alpha, const scalar_type_of_t<A>> and
    (dynamic_rows<A> or dynamic_rows<U> or row_dimension_of<A>::value == row_dimension_of<U>::value) and
    (has_dynamic_dimensions<A> or square_matrix<A>), int> = 0>
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
    (triangular_matrix<A> or self_adjoint_matrix<A> or (zero_matrix<A> and has_dynamic_dimensions<A>) or
      (constant_matrix<A> and not complex_number<scalar_type_of<A>> and has_dynamic_dimensions<A>) or
      (dynamic_rows<A> and dynamic_columns<A>) or (dynamic_rows<A> and column_vector<A>) or
      (dynamic_columns<A> and row_vector<A>)) and
    (dynamic_rows<U> or dynamic_rows<A> or row_dimension_of_v<U> == row_dimension_of_v<A>) and
    (has_dynamic_dimensions<A> or square_matrix<A>)
#else
  template<typename A, typename U, typename Alpha, std::enable_if_t<
    std::is_convertible_v<Alpha, const scalar_type_of_t<A>> and
    (triangular_matrix<A> or self_adjoint_matrix<A> or (zero_matrix<A> and has_dynamic_dimensions<A>) or
      (constant_matrix<A> and not complex_number<scalar_type_of<A>> and has_dynamic_dimensions<A>) or
      (dynamic_rows<A> and dynamic_columns<A>) or (dynamic_rows<A> and column_vector<A>) or
      (dynamic_columns<A> and row_vector<A>)) and
    (dynamic_rows<U> or dynamic_rows<A> or row_dimension_of<U>::value == row_dimension_of<A>::value) and
    (has_dynamic_dimensions<A> or square_matrix<A>), int> = 0>
#endif
  inline decltype(auto)
  rank_update(A&& a, U&& u, Alpha alpha = 1)
  {
    if constexpr (dynamic_rows<A> or dynamic_rows<U>) if (get_dimensions_of<0>(a) != get_dimensions_of<0>(u))
      throw std::domain_error {
        "In rank_update, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) + ") do not match rows of u (" +
        std::to_string(get_index_dimension_of<0>(u)) + ")"};

    if constexpr (triangular_matrix<A>)
    {
      constexpr TriangleType t = triangle_type_of_v<A>;
      return rank_update_triangular<t>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
    else if constexpr (zero_matrix<A> and has_dynamic_dimensions<A>)
    {
      return rank_update_triangular<TriangleType::diagonal>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
    else if constexpr (self_adjoint_matrix<A>)
    {
      constexpr TriangleType t = self_adjoint_triangle_type_of_v<A>;
      return rank_update_self_adjoint<t>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
    else if constexpr (constant_matrix<A> and not complex_number<scalar_type_of<A>> and has_dynamic_dimensions<A>)
    {
      return rank_update_self_adjoint<TriangleType::lower>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
    else
    {
      if constexpr (has_dynamic_dimensions<A>)
        if ((dynamic_rows<A> and get_index_dimension_of<0>(a) != 1) or (dynamic_columns<A> and get_index_dimension_of<1>(a) != 1))
          throw std::domain_error {
            "Non hermitian, non-triangular argument to rank_update expected to be one-by-one, but instead it has " +
            std::to_string(get_index_dimension_of<0>(a)) + " rows and " + std::to_string(get_index_dimension_of<1>(a)) + " columns"};

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
      if (get_dimensions_of<0>(a) != get_dimensions_of<0>(b))
        throw std::domain_error {"The rows of the two operands of the solve function must be the same, but instead "
          "the first operand has " + std::to_string(get_index_dimension_of<0>(a)) + " rows and the second operand has " +
          std::to_string(get_index_dimension_of<0>(b)) + " rows"};
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
    (not constant_matrix<A> or not constant_diagonal_matrix<B> or has_dynamic_dimensions<A> or
      (row_dimension_of_v<A> <= column_dimension_of_v<A> and row_dimension_of_v<B> <= column_dimension_of_v<A>) or
      (row_dimension_of_v<A> == 1 and row_dimension_of_v<B> == 1) or not must_be_exact)
  #else
  template<bool must_be_unique = false, bool must_be_exact = false, typename A, typename B, std::enable_if_t<
    (dynamic_rows<A> or dynamic_rows<B> or row_dimension_of_v<A> == row_dimension_of_v<B>) and
    (not zero_matrix<A> or not zero_matrix<B> or not must_be_unique) and
    (not zero_matrix<A> or not (constant_matrix<B> or constant_diagonal_matrix<B>) or zero_matrix<B> or not must_be_exact) and
    (not constant_matrix<A> or not constant_diagonal_matrix<B> or has_dynamic_dimensions<A> or
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
          return make_zero_matrix_like<B, Scalar>(get_dimensions_of<1>(a), get_dimensions_of<1>(b));
      }
      else
        return make_zero_matrix_like<B, Scalar>(get_dimensions_of<1>(a), get_dimensions_of<1>(b));
    }
    else if constexpr (zero_matrix<A>) //< This will be a non-exact solution unless b is zero.
    {
      if constexpr (dynamic_rows<A> or dynamic_rows<B>) detail::solve_check_A_and_B_rows_match(a, b);
        return make_zero_matrix_like<B, Scalar>(get_dimensions_of<1>(a), get_dimensions_of<1>(b));
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
          auto a_runtime_cols = get_index_dimension_of<1>(a);
          auto c = static_cast<Scalar>(b_const) / (a_runtime_cols * a_const);
          return make_self_contained(c * make_constant_matrix_like<B, 1, Scalar>(Dimensions{a_runtime_cols}, get_dimensions_of<1>(b)));
        }
        else
        {
  #if __cpp_nontype_template_args >= 201911L
          constexpr auto c = static_cast<Scalar>(b_const) / (a_cols * a_const);
          return make_constant_matrix_like<B, c, Scalar>(Dimensions<a_cols>{}, get_dimensions_of<1>(b));
  #else
          if constexpr(b_const % (a_cols * a_const) == 0)
          {
            constexpr std::size_t c = static_cast<std::size_t>(b_const) / (a_cols * static_cast<std::size_t>(a_const));
            return make_constant_matrix_like<B, c, Scalar>(Dimensions<a_cols>{}, get_dimensions_of<1>(b));
          }
          else
          {
            auto c = static_cast<Scalar>(b_const) / (a_cols * a_const);
            auto m = make_constant_matrix_like<B, 1, Scalar>(Dimensions<a_cols>{}, get_dimensions_of<1>(b));
            return make_self_contained(c * to_native_matrix<B>(m));
          }
  #endif
        }
      }
      else if constexpr (row_dimension_of_v<A> == 1 or row_dimension_of_v<B> == 1 or
        (not must_be_exact and (not must_be_unique or
          (not has_dynamic_dimensions<A> and row_dimension_of_v<A> >= column_dimension_of_v<A>))))
      {
        if constexpr (dynamic_rows<A> or dynamic_rows<B>) detail::solve_check_A_and_B_rows_match(a, b);
        return make_self_contained(b / (get_index_dimension_of<1>(a) * constant_coefficient_v<A>));
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
      return n_ary_operation(
        get_all_dimensions_of(b), std::move(op), std::forward<B>(b), diagonal_of(std::forward<A>(a)));
    }
    else
    {
      return interface::LinearAlgebra<std::decay_t<A>>::template solve<must_be_unique, must_be_exact>(
        std::forward<A>(a), std::forward<B>(b));
    }
  }


  /**
   * \brief Perform an LQ decomposition of matrix A=[L,0]Q, L is a lower-triangular matrix, and Q is orthogonal.
   * \tparam A The matrix to be decomposed
   * \returns L as a \ref lower_triangular_matrix
   */
#ifdef __cpp_concepts
  template<indexible A> requires (not euclidean_transformed<A>)
#else
  template<typename A, std::enable_if_t<indexible<A> and (not euclidean_transformed<A>), int> = 0>
#endif
  constexpr auto
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
        if constexpr (dynamic_dimension<A, 1>) return std::sqrt(static_cast<Scalar>(get_index_dimension_of<1>(a)));
        else return OpenKalman::internal::constexpr_sqrt(static_cast<Scalar>(index_dimension_of_v<A, 1>));
      }(std::forward<A>(a));

      if constexpr (dynamic_dimension<A, 0>)
      {
        auto dim = Dimensions {get_index_dimension_of<0>(a)};
        auto col1 = elem * make_constant_matrix_like<A, 1>(dim, Dimensions<1>{});

        auto ret = make_default_dense_writable_matrix_like<A>(dim, dim);

        if (get_dimension_size_of(dim) == 1)
          ret = std::move(col1);
        else
          ret = concatenate_horizontal(std::move(col1), make_zero_matrix_like<A>(dim, dim - Dimensions<1>{}));

        return SquareRootCovariance {std::move(ret), get_dimensions_of<0>(a)};
      }
      else
      {
        auto m = [](Scalar elem){
          constexpr auto dim = index_dimension_of_v<A, 0>;
          auto col1 = elem * make_constant_matrix_like<A, 1>(Dimensions<dim>{}, Dimensions<1>{});
          if constexpr (dimension_size_of_v<dim> == 1) return col1;
          else return concatenate_horizontal(std::move(col1), make_zero_matrix_like<A>(Dimensions<dim>{}, Dimensions<dim - 1>{}));
        }(elem);

        auto ret = Eigen3::TriangularMatrix<decltype(m), TriangleType::lower> {std::move(m)};

        using C = coefficient_types_of_t<A, 0>;
        if constexpr (euclidean_index_descriptor<C>) return ret;
        else return SquareRootCovariance {std::move(ret), C{}};
      }
    }
    else
    {
      auto m = interface::LinearAlgebra<std::decay_t<A>>::LQ_decomposition(std::forward<A>(a));
      auto ret = Eigen3::TriangularMatrix<decltype(m), TriangleType::lower> {std::move(m)};

      // \todo remove the "and false" once SquareRootCovariance works:
      if constexpr (dynamic_dimension<A, 0> and false)
      {
        return SquareRootCovariance {std::move(ret), get_dimensions_of<0>(a)};
      }
      else
      {
        using C = coefficient_types_of_t<A, 0>;
        if constexpr (euclidean_index_descriptor<C>) return ret;
        else return SquareRootCovariance {std::move(ret), C{}};
      }
    }
  }


  /**
   * \brief Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
   * \tparam A The matrix to be decomposed
   * \returns U as an \ref upper_triangular_matrix
   */
#ifdef __cpp_concepts
  template<indexible A>
#else
  template<typename A, std::enable_if_t<indexible<A>, int> = 0>
#endif
  constexpr auto
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
        if constexpr (dynamic_dimension<A, 0>) return std::sqrt(static_cast<Scalar>(get_index_dimension_of<0>(a)));
        else return OpenKalman::internal::constexpr_sqrt(static_cast<Scalar>(index_dimension_of_v<A, 0>));
      }(std::forward<A>(a));

      if constexpr (dynamic_dimension<A, 1>)
      {
        auto dim = Dimensions {get_index_dimension_of<1>(a)};
        auto row1 = elem * make_constant_matrix_like<A, 1>(Dimensions<1>{}, dim);

        auto ret = make_default_dense_writable_matrix_like<A>(dim, dim);

        if (get_dimension_size_of(dim) == 1)
          ret = std::move(row1);
        else
          ret = concatenate_vertical(std::move(row1), make_zero_matrix_like<A>(dim - Dimensions<1>{}, dim));

        return SquareRootCovariance {std::move(ret), get_dimensions_of<1>(a)};
      }
      else
      {
        auto m = [](Scalar elem){
          constexpr auto dim = index_dimension_of_v<A, 1>;
          auto row1 = elem * make_constant_matrix_like<A, 1>(Dimensions<1>{}, Dimensions<dim>{});
          if constexpr (dimension_size_of_v<dim> == 1) return row1;
          else return concatenate_vertical(std::move(row1), make_zero_matrix_like<A>(Dimensions<dim - 1>{}, Dimensions<dim>{}));
        }(elem);

        auto ret = Eigen3::TriangularMatrix<decltype(m), TriangleType::upper> {std::move(m)};

        using C = coefficient_types_of_t<A, 1>;
        if constexpr (euclidean_index_descriptor<C>) return ret;
        else return SquareRootCovariance {std::move(ret), C{}};
      }
    }
    else
    {
      auto m = interface::LinearAlgebra<std::decay_t<A>>::QR_decomposition(std::forward<A>(a));
      auto ret = Eigen3::TriangularMatrix<decltype(m), TriangleType::upper> {std::move(m)};

      // \todo remove the "and false" once SquareRootCovariance works:
      if constexpr (dynamic_dimension<A, 1> and false)
      {
        return SquareRootCovariance {std::move(ret), get_dimensions_of<1>(a)};
      }
      else
      {
        using C = coefficient_types_of_t<A, 1>;
        if constexpr (euclidean_index_descriptor<C>) return ret;
        else return SquareRootCovariance {std::move(ret), C{}};
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
