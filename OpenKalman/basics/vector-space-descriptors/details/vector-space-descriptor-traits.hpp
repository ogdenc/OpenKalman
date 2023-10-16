/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Forward definitions for \ref vector_space_descriptor.
 */

#ifndef OPENKALMAN_INDEX_DESCRIPTOR_TRAITS_HPP
#define OPENKALMAN_INDEX_DESCRIPTOR_TRAITS_HPP

#include <type_traits>

namespace OpenKalman
{
  // ------------------------------ //
  //   composite_vector_space_descriptor   //
  // ------------------------------ //

  namespace detail
  {
    template<typename T>
    struct is_composite_vector_space_descriptor : std::false_type {};

    template<typename...C>
    struct is_composite_vector_space_descriptor<TypedIndex<C...>> : std::true_type {};

    template<typename...AllowableScalarTypes>
    struct is_composite_vector_space_descriptor<DynamicTypedIndex<AllowableScalarTypes...>> : std::true_type {};
  }


  /**
   * \brief T is a \ref composite_vector_space_descriptor.
   * \details A composite \ref vector_space_descriptor object is a container for other \ref vector_space_descriptor, and can either be
   * TypedIndex or DynamicTypedIndex.
   * \sa TypedIndex, DynamicTypedIndex.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept composite_vector_space_descriptor =
#else
  constexpr bool composite_vector_space_descriptor =
#endif
    vector_space_descriptor<T> and detail::is_composite_vector_space_descriptor<std::decay_t<T>>::value;


  // --------------------------------- //
  //   atomic_fixed_vector_space_descriptor   //
  // --------------------------------- //

  /**
   * \brief T is an atomic (non-separable or non-composite) group of fixed \ref vector_space_descriptor.
   * \details These \ref vector_space_descriptor objects are suitable for incorporation in \ref composite_vector_space_descriptor.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept atomic_fixed_vector_space_descriptor =
#else
  constexpr bool atomic_fixed_vector_space_descriptor =
#endif
    fixed_vector_space_descriptor<T> and (not composite_vector_space_descriptor<T>);


  // ------------------------------------- //
  //   replicate_fixed_vector_space_descriptor   //
  // ------------------------------------- //

  /**
   * \brief Replicate a set of \ref vector_space_descriptor a given number of times.
   * \tparam C A \ref vector_space_descriptor object to be repeated.
   * \tparam N The number of times to repeat coefficient C.
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor C, std::size_t N> requires (N != dynamic_size)
#else
  template<typename C, std::size_t N>
#endif
  struct replicate_fixed_vector_space_descriptor
  {
  private:

#ifndef __cpp_concepts
    static_assert(fixed_vector_space_descriptor<C>);
    static_assert(N != dynamic_size);
#endif

    template<typename T, std::size_t...I>
    static constexpr auto replicate_inds(std::index_sequence<I...>)
    {
      return TypedIndex<std::conditional_t<(I==I), T, T>...> {};
    };

  public:

    using type = std::conditional_t<N == 1, C, decltype(replicate_inds<std::decay_t<C>>(std::make_index_sequence<N> {}))>;
  };


  /**
   * \brief Helper template for \ref replicate_fixed_vector_space_descriptor.
   */
  template<typename C, std::size_t N>
  using replicate_fixed_vector_space_descriptor_t = typename replicate_fixed_vector_space_descriptor<C, N>::type;


  // -------------------------------------- //
  //   concatenate_fixed_vector_space_descriptor   //
  // -------------------------------------- //

  template<>
  struct concatenate_fixed_vector_space_descriptor<>
  {
    using type = TypedIndex<>;
  };

  template<typename C, typename...Cs>
  struct concatenate_fixed_vector_space_descriptor<C, Cs...>
  {
    using type = typename concatenate_fixed_vector_space_descriptor<Cs...>::type::template Prepend<C>;
  };

  template<typename...C, typename...Cs>
  struct concatenate_fixed_vector_space_descriptor<TypedIndex<C...>, Cs...>
  {
    using type = typename concatenate_fixed_vector_space_descriptor<Cs...>::type::template Prepend<C...>;
  };


  // ------------------------------------ //
  //   canonical_fixed_vector_space_descriptor   //
  // ------------------------------------ //

#ifdef __cpp_concepts
  template<atomic_fixed_vector_space_descriptor C> requires (not euclidean_vector_space_descriptor<C>)
  struct canonical_fixed_vector_space_descriptor<C>
#else
  template<typename C>
  struct canonical_fixed_vector_space_descriptor<C, std::enable_if_t<
    atomic_fixed_vector_space_descriptor<C> and (not euclidean_vector_space_descriptor<C>)>>
#endif
  {
    using type = TypedIndex<C>;
  };


#ifdef __cpp_concepts
  template<atomic_fixed_vector_space_descriptor C> requires (euclidean_vector_space_descriptor<C>)
  struct canonical_fixed_vector_space_descriptor<C>
#else
  template<typename C>
  struct canonical_fixed_vector_space_descriptor<C, std::enable_if_t<
    atomic_fixed_vector_space_descriptor<C> and (euclidean_vector_space_descriptor<C>)>>
#endif
  {
    using type = std::conditional_t<
      dimension_size_of_v<C> == 1,
      TypedIndex<replicate_fixed_vector_space_descriptor_t<Dimensions<1>, dimension_size_of_v<C>>>,
      replicate_fixed_vector_space_descriptor_t<Dimensions<1>, dimension_size_of_v<C>>>;
  };


  template<typename...Cs>
  struct canonical_fixed_vector_space_descriptor<TypedIndex<TypedIndex<Cs...>>>
  {
    using type = typename canonical_fixed_vector_space_descriptor<TypedIndex<Cs...>>::type;
  };


  template<typename...Cs>
  struct canonical_fixed_vector_space_descriptor<TypedIndex<Cs...>>
  {
    using type = concatenate_fixed_vector_space_descriptor_t<typename canonical_fixed_vector_space_descriptor<Cs>::type...>;
  };


  // ----------------------- //
  //   maybe_equivalent_to   //
  // ----------------------- //

  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T, typename U>
#else
    template<typename T, typename U, typename = void>
#endif
    struct is_maybe_equivalent_to_impl : std::true_type {};


#ifdef __cpp_concepts
    template<fixed_vector_space_descriptor T, fixed_vector_space_descriptor U>
    struct is_maybe_equivalent_to_impl<T, U>
#else
    template<typename T, typename U>
    struct is_maybe_equivalent_to_impl<T, U, std::enable_if_t<fixed_vector_space_descriptor<T> and fixed_vector_space_descriptor<U>>>
#endif
      : std::bool_constant<std::is_same_v<canonical_fixed_vector_space_descriptor_t<T>, canonical_fixed_vector_space_descriptor_t<U>>> {};


    template<typename...Ts>
    struct is_maybe_equivalent_to : std::true_type {};

    template<typename T, typename...Ts>
    struct is_maybe_equivalent_to<T, Ts...> : std::bool_constant<
      dynamic_vector_space_descriptor<T> ?
      is_maybe_equivalent_to<Ts...>::value :
      (... and is_maybe_equivalent_to_impl<T, Ts>::value)> {};
  }


  /**
   * \brief Specifies that a set of \ref vector_space_descriptor objects may be equivalent based on what is known at compile time.
   * \details Sets of coefficients are equivalent if they are treated functionally the same.
   * - Any coefficient or group of coefficients is equivalent to itself.
   * - TypedIndex<As...> is equivalent to TypedIndex<Bs...>, if each As is equivalent to its respective Bs.
   * - TypedIndex<A> is equivalent to A, and vice versa.
   * \par Example:
   * <code>equivalent_to&lt;Axis, TypedIndex&lt;Axis&gt;&gt;</code>
   */
  template<typename...Ts>
#ifdef __cpp_concepts
  concept maybe_equivalent_to =
#else
  constexpr bool maybe_equivalent_to =
#endif
    detail::is_maybe_equivalent_to<Ts...>::value;


  // ----------------- //
  //   equivalent_to   //
  // ----------------- //

  /**
   * \brief Specifies that a set of \ref vector_space_descriptor objects are known at compile time to be equivalent.
   * \details Sets of coefficients are equivalent if they are treated functionally the same.
   * - Any coefficient or group of coefficients is equivalent to itself.
   * - TypedIndex<As...> is equivalent to TypedIndex<Bs...>, if each As is equivalent to its respective Bs.
   * - TypedIndex<A> is equivalent to A, and vice versa.
   * \par Example:
   * <code>equivalent_to&lt;Axis, TypedIndex&lt;Axis&gt;&gt;</code>
   */
  template<typename...Ts>
#ifdef __cpp_concepts
  concept equivalent_to =
#else
  constexpr bool equivalent_to =
#endif
    (fixed_vector_space_descriptor<Ts> and ...) and maybe_equivalent_to<Ts...>;


  // ------------- //
  //   prefix_of   //
  // ------------- //

  namespace detail
  {
    /**
     * \internal
     * \brief Type trait testing whether T (a set of coefficients) is a prefix of U.
     * \details If T is a prefix of U, then U is equivalent_to concatenating T with the remaining part of U.
     */
#ifdef __cpp_concepts
    template<typename T, typename U>
#else
    template<typename T, typename U, typename = void>
#endif
    struct is_prefix_of : std::false_type {};


#ifdef __cpp_concepts
    template<typename C1, typename C2> requires equivalent_to<C1, C2>
    struct is_prefix_of<C1, C2>
#else
    template<typename C1, typename C2>
    struct is_prefix_of<C1, C2, std::enable_if_t<equivalent_to<C1, C2>>>
#endif
      : std::true_type {};


#ifdef __cpp_concepts
    template<typename C>
    struct is_prefix_of<TypedIndex<>, C>
#else
    template<typename C>
    struct is_prefix_of<TypedIndex<>, C, std::enable_if_t<not equivalent_to<TypedIndex<>, C>>>
#endif
      : std::true_type {};


    template<typename C1, typename...Cs>
    struct is_prefix_of<C1, TypedIndex<C1, Cs...>> : std::true_type {};


#ifdef __cpp_concepts
    template<typename C, typename...C1, typename...C2>
    struct is_prefix_of<TypedIndex<C, C1...>, TypedIndex<C, C2...>>
#else
    template<typename C, typename...C1, typename...C2>
    struct is_prefix_of<TypedIndex<C, C1...>, TypedIndex<C, C2...>, std::enable_if_t<
      (not equivalent_to<TypedIndex<C, C1...>, TypedIndex<C, C2...>>)>>
#endif
      : std::bool_constant<is_prefix_of<TypedIndex<C1...>, TypedIndex<C2...>>::value> {};

  } // namespace detail


  /**
   * \brief T is a prefix of U, where T and U are sets of coefficients.
   * \details If T is a prefix of U, then U is equivalent_to concatenating T with the remaining part of U.
   * C is a prefix of TypedIndex<C, Cs...> for any typed \ref vector_space_descriptor Cs.
   * T is a prefix of U if equivalent_to<T, U>.
   * TypedIndex<> is a prefix of any set of coefficients.
   * \par Example:
   * <code>prefix_of&lt;TypedIndex&lt;Axis&gt;, TypedIndex&lt;Axis, angle::Radians&gt;&gt;</code>
   */
  template<typename T, typename U>
#ifdef __cpp_concepts
  concept prefix_of =
#else
  constexpr bool prefix_of =
#endif
    fixed_vector_space_descriptor<T> and fixed_vector_space_descriptor<U> and detail::is_prefix_of<
      canonical_fixed_vector_space_descriptor_t<std::decay_t<T>>, canonical_fixed_vector_space_descriptor_t<std::decay_t<U>>>::value;


  // ----------- //
  //   head_of   //
  // ----------- //

  namespace detail
  {
  #ifdef __cpp_concepts
    template<typename T>
  #else
    template<typename T, typename = void>
  #endif
    struct head_tail_id_split;


    template<>
    struct head_tail_id_split<TypedIndex<>>  { using head = TypedIndex<>; using tail = TypedIndex<>; };


    template<typename C>
    struct head_tail_id_split<TypedIndex<C>> { using head = C; using tail = TypedIndex<>; };


    template<typename C0, typename...Cs>
    struct head_tail_id_split<TypedIndex<C0, Cs...>> { using head = C0; using tail = TypedIndex<Cs...>; };


  #ifdef __cpp_concepts
    template<atomic_fixed_vector_space_descriptor C>
    struct head_tail_id_split<C>
  #else
    template<typename C>
    struct head_tail_id_split<C, std::enable_if_t<atomic_fixed_vector_space_descriptor<C>>>
  #endif
    { using head = C; using tail = TypedIndex<>; };

  } // namespace detail


  /**
   * \brief Type trait extracting the head of a \ref fixed_vector_space_descriptor.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct head_of;


#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor T>
  struct head_of<T>
#else
  template<typename T>
  struct head_of<T, std::enable_if_t<fixed_vector_space_descriptor<T>>>
#endif
    { using type = typename detail::head_tail_id_split<canonical_fixed_vector_space_descriptor_t<T>>::head; };


  /**
   * \brief Helper for \ref head_of.
   */
  template<typename T>
  using head_of_t = typename head_of<T>::type;


  // ----------- //
  //   tail_of   //
  // ----------- //

  /**
   * \brief Type trait extracting the tail of a \ref fixed_vector_space_descriptor.
   */
  #ifdef __cpp_concepts
  template<typename T>
  #else
  template<typename T, typename = void>
  #endif
  struct tail_of ;


#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor T>
  struct tail_of<T>
#else
  template<typename T>
  struct tail_of<T, std::enable_if_t<fixed_vector_space_descriptor<T>>>
#endif
    { using type = typename detail::head_tail_id_split<canonical_fixed_vector_space_descriptor_t<T>>::tail; };


  /**
   * \brief Helper for \ref tail_of.
   */
  template<typename T>
  using tail_of_t = typename tail_of<T>::type;


  // -------------------------------------------------------------------------------------------------- //
  //   has_uniform_dimension_type, uniform_dimension_type_of, equivalent_to_uniform_dimension_type_of   //
  // -------------------------------------------------------------------------------------------------- //

  namespace detail
  {
#ifdef __cpp_concepts
    template<typename C>
#else
    template<typename C, typename = void>
#endif
    struct uniform_dimension_impl : std::false_type {};


#ifdef __cpp_concepts
    template<atomic_fixed_vector_space_descriptor C> requires (dimension_size_of_v<C> == 1)
    struct uniform_dimension_impl<C>
#else
    template<typename C>
    struct uniform_dimension_impl<C, std::enable_if_t<atomic_fixed_vector_space_descriptor<C> and (dimension_size_of_v<C> == 1)>>
#endif
      : std::true_type { using uniform_type = C; };


#ifdef __cpp_concepts
    template<typename C> requires (dimension_size_of_v<C> == 1)
    struct uniform_dimension_impl<TypedIndex<C>>
#else
    template<typename C>
    struct uniform_dimension_impl<TypedIndex<C>, std::enable_if_t<dimension_size_of_v<C> == 1>>
#endif
      : uniform_dimension_impl<C> {};


#ifdef __cpp_concepts
    template<atomic_fixed_vector_space_descriptor C, fixed_vector_space_descriptor...Cs> requires (dimension_size_of_v<C> == 1) and
      (sizeof...(Cs) > 0) and std::same_as<C, typename uniform_dimension_impl<TypedIndex<Cs...>>::uniform_type>
    struct uniform_dimension_impl<TypedIndex<C, Cs...>>
#else
    template<typename C, typename...Cs>
    struct uniform_dimension_impl<TypedIndex<C, Cs...>, std::enable_if_t<
      atomic_fixed_vector_space_descriptor<C> and (... and fixed_vector_space_descriptor<Cs>) and (dimension_size_of_v<C> == 1) and
        (sizeof...(Cs) > 0) and std::is_same<C, typename uniform_dimension_impl<TypedIndex<Cs...>>::uniform_type>::value>>
#endif
      : std::true_type { using uniform_type = C; };


#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct uniform_dimension_impl_17 : std::false_type {};

    template<typename T>
    struct uniform_dimension_impl_17<T, std::void_t<typename canonical_fixed_vector_space_descriptor<T>::type>>
      : detail::uniform_dimension_impl<canonical_fixed_vector_space_descriptor_t<std::decay_t<T>>> {};
#endif

  } // namespace detail


  /**
   * \brief T is a \ref fexed_vector_space_descriptor object comprising a uniform set of 1D \ref atomic_fixed_vector_space_descriptor types.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept has_uniform_dimension_type = fixed_vector_space_descriptor<T> and
    (euclidean_vector_space_descriptor<T> or detail::uniform_dimension_impl<canonical_fixed_vector_space_descriptor_t<std::decay_t<T>>>::value);
#else
  constexpr bool has_uniform_dimension_type = fixed_vector_space_descriptor<T> and
    (euclidean_vector_space_descriptor<T> or detail::uniform_dimension_impl_17<T>::value);
#endif


  /**
   * \brief If T \ref has_uniform_dimension_type, member <code>type</code> is an alias for that type.
   * \sa uniform_dimension_type_of_t
   */
#ifdef __cpp_concepts
  template<has_uniform_dimension_type T>
  struct uniform_dimension_type_of
#else
  template<typename T, typename Enable = void>
  struct uniform_dimension_type_of {};

  template<typename T>
  struct uniform_dimension_type_of<T, std::enable_if_t<has_uniform_dimension_type<T>>>
#endif
  {
    using type = typename detail::uniform_dimension_impl<canonical_fixed_vector_space_descriptor_t<std::decay_t<T>>>::uniform_type;
  };


  /**
   * \brief Helper template for \ref uniform_dimension_type_of.
   */
#ifdef __cpp_concepts
  template<has_uniform_dimension_type T>
#else
  template<typename T>
#endif
  using uniform_dimension_type_of_t = typename uniform_dimension_type_of<T>::type;


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename C, typename = void>
    struct equivalent_to_uniform_dimension_type_of_impl : std::false_type {};

    template<typename T, typename C>
    struct equivalent_to_uniform_dimension_type_of_impl<T, C, std::enable_if_t<
      equivalent_to<T, typename uniform_dimension_type_of<C>::type>>> : std::true_type {};
  }
#endif


  /**
   * \brief T is equivalent to the uniform dimension type of C.
   * \tparam T A 1D \ref atomic_fixed_vector_space_descriptor
   * \tparam C a \ref has_uniform_dimension_type
   */
  template<typename T, typename C>
#ifdef __cpp_concepts
  concept equivalent_to_uniform_dimension_type_of = equivalent_to<T, uniform_dimension_type_of_t<C>>;
#else
  constexpr bool equivalent_to_uniform_dimension_type_of = detail::equivalent_to_uniform_dimension_type_of_impl<T, C>::value;
#endif

} // namespace OpenKalman

#endif //OPENKALMAN_INDEX_DESCRIPTOR_TRAITS_HPP
