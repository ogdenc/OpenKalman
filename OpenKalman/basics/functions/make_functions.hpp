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
 * \brief Overloaded general functions for making math objects.
 */

#ifndef OPENKALMAN_MAKE_FUNCTIONS_HPP
#define OPENKALMAN_MAKE_FUNCTIONS_HPP

namespace OpenKalman
{
  using namespace interface;

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
  constexpr writable auto
#else
  template<typename T, typename Scalar = scalar_type_of_t<T>, typename...D, std::enable_if_t<
    indexible<T> and scalar_type<Scalar> and (index_descriptor<D> and ...) and
    (sizeof...(D) == max_indices_of_v<T> or (sizeof...(D) == 0 and not has_dynamic_dimensions<T>)), int> = 0>
  constexpr auto
#endif
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
   * \tparam Scalar A scalar type for the new matrix.
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar, indexible T>
  constexpr writable auto
#else
  template<typename Scalar, typename T, std::enable_if_t<scalar_type<Scalar> and indexible<T>, int> = 0>
  constexpr auto
#endif
  make_default_dense_writable_matrix_like(const T& t)
  {
    if constexpr (writable<T> and std::is_same_v<Scalar, scalar_type_of_t<T>>)
    {
      return t;
    }
    else
    {
      using NewScalar = std::conditional_t<std::is_void_v<Scalar>, scalar_type_of_t<T>, Scalar>;
      return std::apply(
        [](auto&&...d) { return make_default_dense_writable_matrix_like<T, NewScalar>(std::forward<decltype(d)>(d)...); },
        get_all_dimensions_of(t));
    }
  }


  /**
   * \overload
   * \brief Make a default, dense, writable matrix based on an existing object.
   * \param t The existing object
   */
#ifdef __cpp_concepts
  template<indexible T>
  constexpr writable auto
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
  constexpr auto
#endif
  make_default_dense_writable_matrix_like(const T& t)
  {
    if constexpr (writable<T>) return t;
    else return make_default_dense_writable_matrix_like<scalar_type_of_t<T>>(t);
  }


  // --------------------------------- //
  //  make_dense_writable_matrix_from  //
  // --------------------------------- //

  /**
   * \brief Convert the argument to a dense, writable matrix.
   * \tparam Scalar The Scalar type of the new matrix, if different than that of Arg
   * \tparam Arg The object from which the new matrix is based
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar, indexible Arg>
  constexpr /*writable*/ decltype(auto)
#else
  template<typename Scalar, typename Arg, std::enable_if_t<scalar_type<Scalar> and indexible<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  make_dense_writable_matrix_from(Arg&& arg)
  {
    if constexpr (writable<Arg> and std::is_same_v<Scalar, scalar_type_of_t<Arg>>)
      return std::forward<Arg>(arg);
    else
      return EquivalentDenseWritableMatrix<std::decay_t<Arg>, std::decay_t<Scalar>>::convert(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Convert the argument to a dense, writable matrix.
   * \tparam Arg The object from which the new matrix is based
   */
#ifdef __cpp_concepts
  template<indexible Arg>
  constexpr /*writable*/ decltype(auto)
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  make_dense_writable_matrix_from(Arg&& arg)
  {
    if constexpr (writable<Arg>)
      return std::forward<Arg>(arg);
    else
      return EquivalentDenseWritableMatrix<std::decay_t<Arg>, scalar_type_of_t<Arg>>::convert(std::forward<Arg>(arg));
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
  inline writable auto
#else
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wdiv-by-zero"
  template<typename M, typename Scalar = scalar_type_of_t<M>, typename...Ds, typename...Args, std::enable_if_t<
    indexible<M> and scalar_type<Scalar> and (index_descriptor<Ds> and ...) and
    (std::is_convertible_v<Args, const Scalar> and ...) and
    (sizeof...(Args) % ((dynamic_index_descriptor<Ds> ? 1 : dimension_size_of_v<Ds>) * ... * 1) == 0), int> = 0>
  inline auto
#endif
  make_dense_writable_matrix_from(const std::tuple<Ds...>& d_tup, Args...args)
  {
    using Trait = EquivalentDenseWritableMatrix<std::decay_t<M>, std::decay_t<Scalar>>;
    using Nat = decltype(Trait::make_default(std::declval<Ds&&>()...));
    return MatrixTraits<std::decay_t<Nat>>::make(static_cast<const Scalar>(args)...);
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
      return make_dense_writable_matrix_from<M, Scalar>(
        std::tuple {[]{
          if constexpr (dynamic_dimension<M, I>) return Dimensions<sizeof...(Args) / dims>{};
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
  template<indexible M, scalar_type Scalar = scalar_type_of_t<M>, std::convertible_to<const Scalar> ... Args>
  requires (detail::check_make_dense_args<M, sizeof...(Args)>())
  inline writable auto
#else
  template<typename M, typename Scalar = scalar_type_of_t<M>, typename ... Args, std::enable_if_t<
    indexible<M> and scalar_type<Scalar> and (std::is_convertible_v<Args, const Scalar> and ...) and
    (detail::check_make_dense_args<M, sizeof...(Args)>()), int> = 0>
  inline auto
#endif
  make_dense_writable_matrix_from(Args...args)
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
  inline decltype(auto)
  to_native_matrix(Arg&& arg)
  {
    return EquivalentDenseWritableMatrix<std::decay_t<T>>::to_native_matrix(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief If it isn't already, convert arg into a native matrix within its library.
   */
#ifdef __cpp_concepts
  template<indexible Arg>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
  inline decltype(auto)
  to_native_matrix(Arg&& arg)
  {
    return to_native_matrix<std::decay_t<Arg>, Arg>(std::forward<Arg>(arg));
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
  nested_matrix(Arg&& arg)
  {
      return Dependencies<std::decay_t<Arg>>::template get_nested_matrix<i>(std::forward<Arg>(arg));
  }


  // --------------------- //
  //  make_self_contained  //
  // --------------------- //

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
   *   decltype(auto) add(Arg1&& arg1, Arg2&& arg2)
   *   {
   *     return make_self_contained<Arg1, Arg2>(arg1 + arg2);
   *   }
   * \endcode
   * \tparam Ts Generally, these will be forwarding-reference arguments to the directly enclosing function. If all of
   * Ts... are lvalue references, Arg is returned without modification (i.e., without any potential eager evaluation).
   * \tparam Arg The potentially non-self-contained argument to be converted
   * \return A self-contained version of Arg (if it is not already self-contained)
   * \todo Return a new class that internalizes any external dependencies
   * \internal \sa interface::Dependencies
   */
#ifdef __cpp_concepts
  template<indexible...Ts, indexible Arg>
  constexpr /*self_contained<Ts...>*/ decltype(auto)
#else
  template<typename...Ts, typename Arg, std::enable_if_t<(indexible<Ts> and ...) and indexible<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  make_self_contained(Arg&& arg) noexcept
  {
    if constexpr (self_contained<Arg, Ts...>)
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
   * \brief Make a constant matrix based on a particular library object
   * \tparam T A matrix or tensor from a particular library.
   * \tparam constant The constant
   * \tparam Scalar An optional scalar type for the new zero matrix. By default, T's scalar type is used.
   * \param D A set of \ref index_descriptor "index descriptors" defining the dimensions of each index.
   * \todo Swap T and constant positions
   */
#ifdef __cpp_concepts
  template<indexible T, auto constant, scalar_type Scalar = scalar_type_of_t<T>, index_descriptor...D> requires
    (sizeof...(D) == max_indices_of_v<T>)
  constexpr constant_matrix auto
#else
  template<typename T, auto constant, typename Scalar = scalar_type_of_t<T>, typename...D, std::enable_if_t<
    indexible<T> and scalar_type<Scalar> and (index_descriptor<D> and ...) and sizeof...(D) == max_indices_of_v<T>, int> = 0>
  constexpr auto
#endif
  make_constant_matrix_like(D&&...d)
  {
    constexpr auto c =
# if __cpp_nontype_template_args >= 201911L
      real_projection(static_cast<Scalar>(constant));
# else
      constant;
# endif

    using Td = std::decay_t<T>;
#ifdef __cpp_concepts
    if constexpr (requires (D&&...d) {
      SingleConstantMatrixTraits<std::decay_t<T>, std::decay_t<Scalar>>::template make_constant_matrix<c>(std::forward<D>(d)...);
    })
#else
    if constexpr (detail::make_constant_matrix_trait_defined<Td, Scalar, constant, void, D...>::value)
#endif
    {
      return SingleConstantMatrixTraits<std::decay_t<T>, std::decay_t<Scalar>>::template make_constant_matrix<c>(std::forward<D>(d)...);
    }
    else
    {
      // Default behavior if interface function not defined:
      using N = std::decay_t<decltype(EquivalentDenseWritableMatrix<Td, std::decay_t<Scalar>>::make_default(std::declval<D&&>()...))>;
      return ConstantAdapter<N, c> {std::forward<D>(d)...};
    }
  }


  /**
   * \overload
   * \brief Make a constant matrix based on a particular library object
   * \tparam T A matrix or tensor from a particular library.
   * \tparam Scalar An optional scalar type for the new zero matrix. By default, T's scalar type is used.
   * \param D A set of \ref index_descriptor "index descriptors" defining the dimensions of each index.
   */
#ifdef __cpp_concepts
  template<indexible T, scalar_type Scalar = scalar_type_of_t<T>, index_descriptor...D> requires
    (sizeof...(D) == max_indices_of_v<T>)
  constexpr constant_matrix auto
#else
  template<typename T, typename Scalar = scalar_type_of_t<T>, typename...D, std::enable_if_t<
    indexible<T> and scalar_type<Scalar> and (index_descriptor<D> and ...) and sizeof...(D) == max_indices_of_v<T>, int> = 0>
  constexpr auto
#endif
  make_constant_matrix_like(D&&...d)
  {
    constexpr auto c = constant_coefficient_v<T>;
# if __cpp_nontype_template_args >= 201911L
    return make_constant_matrix_like<T, c, Scalar>(std::forward<D>(d)...);
# else
    constexpr auto c_integral = static_cast<std::intmax_t>(c);
    static_assert(are_within_tolerance(c, static_cast<Scalar>(c_integral)));
    return make_constant_matrix_like<T, c_integral, Scalar>(std::forward<D>(d)...);
# endif
  }


  /**
   * \overload
   * \brief Make a single-constant matrix based on T, but specifying a new constant and a new scalar type.
   * \tparam T The matrix or array on which the new matrix is patterned.
   * \tparam constant The constant.
   * \tparam Scalar A scalar type for the new matrix.
   */
#ifdef __cpp_concepts
  template<auto constant, scalar_type Scalar, indexible T>
  constexpr constant_matrix auto
#else
  template<auto constant, typename Scalar, typename T, std::enable_if_t<scalar_type<Scalar> and indexible<T>, int> = 0>
  constexpr auto
#endif
  make_constant_matrix_like(const T& t)
  {
    constexpr bool constants_match = []{
      if constexpr (constant_matrix<T>) return are_within_tolerance(constant, constant_coefficient_v<T>);
      else return false;
    }();

    if constexpr (constants_match and std::is_same_v<Scalar, scalar_type_of_t<T>>)
      return t;
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
  constexpr constant_matrix auto
#else
  template<auto constant, typename T, std::enable_if_t<indexible<T>, int> = 0>
  constexpr auto
#endif
  make_constant_matrix_like(const T& t)
  {
    return make_constant_matrix_like<constant, scalar_type_of_t<T>>(t);
  }


  /**
   * \overload
   * \brief Make a single-constant matrix based on a constant T.
   * \tparam T The matrix or array on which the new matrix is patterned.
   * \tparam constant The constant.
   */
#ifdef __cpp_concepts
  template<constant_matrix T>
  constexpr constant_matrix auto
#else
  template<typename T, std::enable_if_t<constant_matrix<T>, int> = 0>
  constexpr auto
#endif
  make_constant_matrix_like(const T& t)
  {
    using Scalar = scalar_type_of_t<T>;
    constexpr auto c = constant_coefficient_v<T>;
# if __cpp_nontype_template_args >= 201911L
    return make_constant_matrix_like<c, Scalar>(t);
# else
    if constexpr (std::is_integral_v<Scalar>) return make_constant_matrix_like<c, Scalar>(t);
    else
    {
      constexpr auto c_integral = static_cast<std::intmax_t>(c);
      static_assert(are_within_tolerance(c, static_cast<Scalar>(c_integral)));
      return make_constant_matrix_like<c_integral, Scalar>(t);
    }
# endif
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
  template<indexible T, auto constant, scalar_type Scalar = scalar_type_of_t<T>> requires (not has_dynamic_dimensions<T>)
  constexpr constant_matrix auto
#else
  template<typename T, auto constant, typename Scalar = scalar_type_of_t<T>, std::enable_if_t<
    indexible<T> and scalar_type<Scalar> and (not has_dynamic_dimensions<T>), int> = 0>
  constexpr auto
#endif
  make_constant_matrix_like()
  {
    return std::apply(
      [](auto&&...arg){ return make_constant_matrix_like<T, constant, Scalar>(std::forward<decltype(arg)>(arg)...); },
      get_all_dimensions_of<T>());
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
  template<indexible T, scalar_type Scalar = scalar_type_of_t<T>> requires (not has_dynamic_dimensions<T>)
  constexpr constant_matrix auto
#else
  template<typename T, typename Scalar = scalar_type_of_t<T>, std::enable_if_t<
    indexible<T> and scalar_type<Scalar> and (not has_dynamic_dimensions<T>), int> = 0>
  constexpr auto
#endif
  make_constant_matrix_like()
  {
    constexpr auto c = constant_coefficient_v<T>;
# if __cpp_nontype_template_args >= 201911L
    return make_constant_matrix_like<T, c, Scalar>();
# else
    if constexpr (std::is_integral_v<Scalar>) return make_constant_matrix_like<T, c, Scalar>();
    else
    {
      constexpr auto c_integral = static_cast<std::intmax_t>(c);
      static_assert(are_within_tolerance(c, static_cast<Scalar>(c_integral)));
      return make_constant_matrix_like<T, c_integral, Scalar>();
    }
# endif
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
   * \brief Make a \ref zero_matrix associated with a particular library.
   * \tparam T A matrix or other tensor within a particular library. Its details are not important.
   * \tparam Scalar An optional scalar type for the new zero matrix. By default, T's scalar type is used.
   * \param D A set of \ref index_descriptor "index descriptors" defining the dimensions of each index.
   */
#ifdef __cpp_concepts
  template<indexible T, scalar_type Scalar = scalar_type_of_t<T>, index_descriptor...D> requires
    (sizeof...(D) == max_indices_of_v<T>)
  constexpr zero_matrix auto
#else
  template<typename T, typename Scalar = scalar_type_of_t<T>, typename...D, std::enable_if_t<indexible<T> and
    scalar_type<Scalar> and (index_descriptor<D> and ...) and sizeof...(D) == max_indices_of_v<T>, int> = 0>
  constexpr auto
#endif
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
      return make_constant_matrix_like<T, 0, Scalar>(std::forward<D>(d)...);
    }
  }


  /**
   * \overload
   * \brief Make a \ref zero_matrix based on an argument, specifying a new scalar type.
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   * \tparam Scalar A scalar type for the new matrix.
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar, indexible T>
  constexpr zero_matrix auto
#else
  template<typename Scalar, typename T, std::enable_if_t<scalar_type<Scalar> and indexible<T>, int> = 0>
  constexpr auto
#endif
  make_zero_matrix_like(const T& t)
  {
    return make_constant_matrix_like<0, Scalar>(t);
  }


  /**
   * \overload
   * \brief Make a zero matrix based on T.
   * \tparam T The matrix or array on which the new matrix is patterned.
   */
#ifdef __cpp_concepts
  template<indexible T>
  constexpr zero_matrix auto
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
  constexpr auto
#endif
  make_zero_matrix_like(const T& t)
  {
    return make_constant_matrix_like<0, scalar_type_of_t<T>>(t);
  }


  /**
   * \overload
   * \brief Make a zero matrix based on the type T, where the dimensions of T are known at compile time.
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   * \tparam Scalar A scalar type for the new matrix (by default, the same as that of T)
   */
#ifdef __cpp_concepts
  template<indexible T, scalar_type Scalar = scalar_type_of_t<T>> requires (not has_dynamic_dimensions<T>)
  constexpr zero_matrix auto
#else
  template<typename T, typename Scalar = scalar_type_of_t<T>, std::enable_if_t<
    indexible<T> and scalar_type<Scalar> and (not has_dynamic_dimensions<T>), int> = 0>
  constexpr auto
#endif
  make_zero_matrix_like()
  {
    return make_constant_matrix_like<T, 0, Scalar>();
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
   * \brief Make an identity matrix based on an object of a particular library.
   * \tparam T The matrix or tensor of a particular library.
   * \tparam Scalar An optional scalar type for the new zero matrix. By default, T's scalar type is used.
   * \param D An \ref index_descriptor "index descriptor" defining the dimensions of each index.
   */
#ifdef __cpp_concepts
  template<indexible T, scalar_type Scalar = scalar_type_of_t<T>, index_descriptor D>
  constexpr identity_matrix auto
#else
  template<typename T, typename Scalar = typename scalar_type_of<T>::type, typename D, std::enable_if_t<
    indexible<T> and scalar_type<Scalar> and index_descriptor<D>, int> = 0>
  constexpr auto
#endif
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
      return DiagonalMatrix {make_constant_matrix_like<Td, 1, Scalar>(std::forward<D>(d), Dimensions<1>{})};
    }
  }


  /**
   * \overload
   * \brief Make an identity matrix independent of any library.
   * \tparam Scalar An optional scalar type for the new zero matrix. By default, T's scalar type is used.
   * \param D An \ref index_descriptor "index descriptor" defining the dimensions of each index.
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar, index_descriptor D>
  constexpr identity_matrix auto
#else
  template<typename Scalar, typename D, std::enable_if_t<scalar_type<Scalar> and index_descriptor<D>, int> = 0>
  constexpr auto
#endif
  make_identity_matrix_like(D&& d)
  {
    return DiagonalMatrix {make_constant_matrix_like<1, Scalar>(std::forward<D>(d), Dimensions<1>{})};
  }


  /**
   * \overload
   * \brief Make an identity matrix based on the argument, specifying a new scalar type.
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   * \tparam Scalar A scalar type for the new matrix.
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar, square_matrix<Likelihood::maybe> T>
  constexpr identity_matrix auto
#else
  template<typename Scalar, typename T, std::enable_if_t<scalar_type<Scalar> and square_matrix<T, Likelihood::maybe>, int> = 0>
  constexpr auto
#endif
  make_identity_matrix_like(const T& t)
  {
    if constexpr (identity_matrix<T> and std::is_same_v<Scalar, scalar_type_of_t<T>>)
    {
      return t;
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
  constexpr identity_matrix auto
#else
  template<typename T, std::enable_if_t<indexible<T> and (has_dynamic_dimensions<T> or square_matrix<T>), int> = 0>
  constexpr auto
#endif
  make_identity_matrix_like(const T& t)
  {
    return make_identity_matrix_like<scalar_type_of_t<T>>(t);
  }


  /**
   * \overload
   * \brief Make an identity matrix based on T, which has fixed size, specifying a new scalar type.
   * \tparam T The matrix or array on which the new zero matrix is patterned.
   * \tparam Scalar A scalar type for the new matrix. The default is the scalar type of T.
   */
#ifdef __cpp_concepts
  template<indexible T, scalar_type Scalar = scalar_type_of_t<T>> requires square_matrix<T>
  constexpr identity_matrix auto
#else
  template<typename T, typename Scalar = scalar_type_of_t<T>, std::enable_if_t<indexible<T> and scalar_type<Scalar> and
    square_matrix<T>, int> = 0>
  constexpr auto
#endif
  make_identity_matrix_like()
  {
    return make_identity_matrix_like<T, Scalar>(Dimensions<index_dimension_of_v<T, 0>>{});
  }


  // ----------------------- //
  //  make_hermitian_matrix  //
  // ----------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, TriangleType t, typename = void>
    struct make_hermitian_adapter_defined: std::false_type {};

    template<typename T, TriangleType t>
    struct make_hermitian_adapter_defined<T, t, std::void_t<
      decltype(HermitianTraits<std::decay_t<T>>::template make_hermitian_adapter<t>(std::declval<T&&>()))>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Wraps a matrix in a \ref hermitian_adapter to create a \ref hermitian_matrix.
   * \note The resulting adapter type is not guaranteed to be adapter_type.
   * \tparam adapter_type The intended \ref TriangleType of the result (lower, upper, or diagonal).
   * \tparam Arg A non-hermitian matrix.
   */
#ifdef __cpp_concepts
  template<TriangleType adapter_type = TriangleType::lower, square_matrix<Likelihood::maybe> Arg>
  requires (adapter_type != TriangleType::none) and (not hermitian_matrix<Arg>)
  constexpr hermitian_matrix decltype(auto)
#else
  template<TriangleType adapter_type = TriangleType::lower, typename Arg, std::enable_if_t<
    (adapter_type != TriangleType::none) and square_matrix<Arg, Likelihood::maybe> and (not hermitian_matrix<Arg>), int> = 0>
  constexpr decltype(auto)
#endif
  make_hermitian_matrix(Arg&& arg)
  {
    if constexpr (hermitian_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      using Traits = HermitianTraits<std::decay_t<Arg>>;
# ifdef __cpp_concepts
      if constexpr (requires (Arg&& arg) { Traits::template make_hermitian_adapter<adapter_type>(std::forward<Arg>(arg)); })
# else
      if constexpr (detail::make_hermitian_adapter_defined<Arg, adapter_type>::value)
# endif
      {
        return Traits::template make_hermitian_adapter<adapter_type>(std::forward<Arg>(arg));
      }
      else
      {
        // Default behavior if interface function not defined:
        using pArg = std::conditional_t<std::is_lvalue_reference_v<Arg>, Arg, std::remove_reference_t<decltype(make_self_contained(arg))>>;
        return SelfAdjointMatrix<pArg, adapter_type> {std::forward<Arg>(arg)};
      }
    }
  }


  // ------------------------ //
  //  make_triangular_matrix  //
  // ------------------------ //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, TriangleType t, typename = void>
    struct make_triangular_matrix_defined: std::false_type {};

    template<typename T, TriangleType t>
    struct make_triangular_matrix_defined<T, t, std::void_t<
      decltype(TriangularTraits<std::decay_t<T>>::template make_triangular_matrix<t>(std::declval<T&&>()))>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Create a \ref triangular_matrix from a general matrix.
   * \tparam t The intended \ref TriangleType of the result.
   * \tparam Arg A general matrix to be made triangular.
   */
#ifdef __cpp_concepts
  template<TriangleType t = TriangleType::lower, indexible Arg> requires
    (t == TriangleType::lower or t == TriangleType::upper) and (square_matrix<Arg> or has_dynamic_dimensions<Arg>)
  constexpr /*triangular_matrix<t>*/ decltype(auto)
#else
  template<TriangleType t = TriangleType::lower, typename Arg, std::enable_if_t<indexible<Arg> and
    (t == TriangleType::lower or t == TriangleType::upper) and (square_matrix<Arg> or has_dynamic_dimensions<Arg>), int> = 0>
  constexpr decltype(auto)
#endif
  make_triangular_matrix(Arg&& arg)
  {
    if constexpr (triangular_matrix<Arg> and (t == triangle_type_of_v<Arg> or diagonal_matrix<Arg>))
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      using Traits = TriangularTraits<std::decay_t<Arg>>;
# ifdef __cpp_concepts
      if constexpr (requires (Arg&& arg) { Traits::template make_triangular_matrix<t>(std::forward<Arg>(arg)); })
# else
      if constexpr (detail::make_triangular_matrix_defined<Arg, t>::value)
# endif
      {
        return Traits::template make_triangular_matrix<t>(std::forward<Arg>(arg));
      }
      else
      {
        // Default behavior if interface function not defined:
        using pArg = std::conditional_t<std::is_lvalue_reference_v<Arg>, Arg, std::remove_reference_t<decltype(make_self_contained(arg))>>;
        return TriangularMatrix<pArg, t> {std::forward<Arg>(arg)};
      }
    }
  }

} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_FUNCTIONS_HPP
