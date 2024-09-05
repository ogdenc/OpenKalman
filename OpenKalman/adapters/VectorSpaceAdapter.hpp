/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_VECTORSPACEADAPTER_HPP
#define OPENKALMAN_VECTORSPACEADAPTER_HPP

namespace OpenKalman
{
#ifdef __cpp_concepts
  template<indexible NestedObject, vector_space_descriptor...Vs>
  requires internal::not_more_fixed_than<NestedObject, Vs...> and (not internal::less_fixed_than<NestedObject, Vs...>) and
    internal::maybe_same_shape_as_vector_space_descriptors<NestedObject, Vs...>
#else
  template<typename NestedObject, typename...Vs>
#endif
  struct VectorSpaceAdapter : internal::AdapterBase<VectorSpaceAdapter<NestedObject, Vs...>, const NestedObject>
  {
  private:

#ifndef __cpp_concepts
    static_assert(indexible<NestedObject>);
    static_assert((... and vector_space_descriptor<Vs>));
    static_assert(internal::not_more_fixed_than<NestedObject, Vs...>);
    static_assert(not internal::less_fixed_than<NestedObject, Vs...>);
    static_assert(internal::maybe_same_shape_as_vector_space_descriptors<NestedObject, Vs...>);
#endif

    using Base = internal::AdapterBase<VectorSpaceAdapter, const NestedObject>;

  public:

    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr VectorSpaceAdapter() requires std::default_initializable<Base> and (... and fixed_vector_space_descriptor<Vs>)
#else
    template<typename B = Base, std::enable_if_t<std::is_default_constructible<B>::value
      and (... and fixed_vector_space_descriptor<Vs>), int> = 0>
    constexpr VectorSpaceAdapter()
#endif
      : Base {}, my_descriptors{} {}


    /**
     * \brief Construct from a compatible indexible object.
     * \tparam Arg An \ref indexible object
     * \tparam Ds A set of \ref vector_space_descriptor objects, which must be \ref equivalent_to those of the VectorSpaceAdapter
     */
#ifdef __cpp_concepts
    template<internal::maybe_same_shape_as_vector_space_descriptors<Vs...> Arg, equivalent_to<Vs>...Ds> requires
      (not internal::vector_space_adapter<Arg>) and std::constructible_from<Base, Arg&&>
#else
    template<typename Arg, typename...Ds, std::enable_if_t<
      internal::maybe_same_shape_as_vector_space_descriptors<Arg, Vs...> and (... and equivalent_to<Ds, Vs>) and
      (not internal::vector_space_adapter<Arg>) and std::is_constructible_v<Base, Arg&&>, int> = 0>
#endif
    constexpr VectorSpaceAdapter(Arg&& arg, Ds&&...ds)
      : Base {std::forward<Arg>(arg)}, my_descriptors {std::forward<Ds>(ds)...} {}


    /**
     * \overload
     * \brief Construct from another VectorSpaceAdapter, but replace the vector space descriptors.
     */
#ifdef __cpp_concepts
    template<internal::maybe_same_shape_as_vector_space_descriptors<Vs...> Arg, equivalent_to<Vs>...Ds> requires
      internal::vector_space_adapter<Arg> and std::constructible_from<Base, nested_object_of_t<Arg&&>>
#else
    template<typename Arg, typename...Ds, std::enable_if_t<
      internal::maybe_same_shape_as_vector_space_descriptors<Arg, Vs...> and (... and equivalent_to<Ds, Vs>) and
      internal::vector_space_adapter<Arg> and std::is_constructible_v<Base, typename nested_object_of<Arg&&>::type>, int> = 0>
#endif
    constexpr VectorSpaceAdapter(Arg&& arg, Ds&&...ds)
      : Base {nested_object(std::forward<Arg>(arg))}, my_descriptors {std::forward<Ds>(ds)...} {}


    /**
     * \brief Assign from another compatible indexible object.
     */
#ifdef __cpp_concepts
    template<internal::maybe_same_shape_as_vector_space_descriptors<Vs...> Arg> requires
      std::assignable_from<std::add_lvalue_reference_t<NestedObject>, Arg&&> and
      requires(Arg&& arg) { {count_indices(arg)} -> static_index_value; } and
      requires(std::tuple<std::decay_t<Vs>...> my_descriptors, Arg&& arg) { my_descriptors = all_vector_space_descriptors(arg); }
#else
    template<typename Arg, std::enable_if_t<internal::maybe_same_shape_as_vector_space_descriptors<Arg, Vs...> and
      std::is_assignable_v<std::add_lvalue_reference_t<NestedObject>, Arg&&> and
      static_index_value<decltype(count_indices(std::declval<Arg&&>()))> and
      std::is_assignable_v<std::tuple<std::decay_t<Vs>...>, decltype(all_vector_space_descriptors(std::declval<Arg&&>()))>, int> = 0>
#endif
    constexpr VectorSpaceAdapter& operator=(Arg&& arg)
    {
      my_descriptors = all_vector_space_descriptors(arg);
      Base::operator=(std::forward<Arg>(arg));
      return *this;
    }


    /// Increment from another VectorSpaceAdapter.
#ifdef __cpp_concepts
    auto& operator+=(const VectorSpaceAdapter& other) requires
      requires(Base& base, const VectorSpaceAdapter& v) { base += v.nested_object(); }
#else
    template<typename B = Base, typename = std::void_t<
      decltype(std::declval<Base&>() += std::declval<const VectorSpaceAdapter&>().nested_object())>>
    auto& operator+=(const VectorSpaceAdapter& other)
#endif
    {
      Base::operator+=(other.nested_object());
      return *this;
    }


    /// Decrement from another VectorSpaceAdapter.
#ifdef __cpp_concepts
    auto& operator-=(const VectorSpaceAdapter& other) requires
      requires(Base& base, const VectorSpaceAdapter& v) { base -= v.nested_object(); }
#else
    template<typename B = Base, typename = std::void_t<
      decltype(std::declval<Base&>() -= std::declval<const VectorSpaceAdapter&>().nested_object())>>
    auto& operator-=(const VectorSpaceAdapter& other)
#endif
    {
      Base::operator-=(other.nested_object());
      return *this;
    }


  protected:

    std::tuple<std::decay_t<Vs>...> my_descriptors;

    friend struct interface::indexible_object_traits<VectorSpaceAdapter>;
    friend struct interface::library_interface<VectorSpaceAdapter>;

  }; // struct VectorSpaceAdapter


  // ------------------------------- //
  //        Deduction Guides         //
  // ------------------------------- //

#ifdef __cpp_concepts
    template<indexible Arg, vector_space_descriptor...Vs> requires (not internal::vector_space_adapter<Arg>)
#else
    template<typename Arg, typename...Vs, std::enable_if_t<indexible<Arg> and (... and vector_space_descriptor<Vs>) and
      (not internal::vector_space_adapter<Arg>), int> = 0>
#endif
    VectorSpaceAdapter(Arg&&, const Vs&...) -> VectorSpaceAdapter<Arg, Vs...>;


#ifdef __cpp_concepts
    template<internal::vector_space_adapter Arg, vector_space_descriptor...Vs>
#else
    template<typename Arg, typename...Vs, std::enable_if_t<
      internal::vector_space_adapter<Arg> and (... and vector_space_descriptor<Vs>), int> = 0>
#endif
    VectorSpaceAdapter(Arg&&, const Vs&...) -> VectorSpaceAdapter<internal::remove_rvalue_reference_t<nested_object_of_t<Arg&&>>, Vs...>;


} // namespace OpenKalman


#endif //OPENKALMAN_VECTORSPACEADAPTER_HPP
