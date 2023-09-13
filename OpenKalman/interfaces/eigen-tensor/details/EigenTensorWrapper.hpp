/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definitions for Eigen3::EigenTensorWrapper
 */

#ifndef OPENKALMAN_EIGENTENSORWRAPPER_HPP
#define OPENKALMAN_EIGENTENSORWRAPPER_HPP

namespace OpenKalman
{
  namespace Eigen3
  {
    template<typename NestedMatrix, typename IndexType>
    struct EigenTensorWrapper : Eigen::TensorBase<EigenTensorWrapper<NestedMatrix, IndexType>, Eigen::ReadOnlyAccessors>
    {
    private:

      using Base = Eigen::TensorBase<EigenTensorWrapper, Eigen::ReadOnlyAccessors>;

      template<typename Arg, std::size_t...is>
      static auto make_dim(const Arg& arg, std::index_sequence<is...>) { return Eigen::Sizes<index_dimension_of_v<Arg, is>...> {}; };

#ifdef __cpp_concepts
      template<has_dynamic_dimensions Arg, std::size_t...is>
#else
      template<typename Arg, std::size_t...is, std::enable_if_t<has_dynamic_dimensions<Arg>, int> = 0>
#endif
      static auto make_dim(const Arg& arg, std::index_sequence<is...>)
      {
        return Eigen::DSizes<IndexType, max_indices_of_v<Arg>> {get_index_dimension_of<is>(arg)...};
      };

    public:

      using Index = IndexType;
      using Scalar = scalar_type_of_t<NestedMatrix>;
      using CoeffReturnType = Scalar;
      using Dimensions = decltype(make_dim(std::declval<NestedMatrix>(), std::make_index_sequence<max_indices_of_v<NestedMatrix>>{}));

      using Nested = EigenTensorWrapper;

      enum {
        IsAligned         = false,
        Layout            = layout_of_v<NestedMatrix> == Layout::right ? Eigen::RowMajor : Eigen::ColMajor,
      };

      /* \internal
       * \brief The return type for coefficient access.
       * \details Depending on whether the object allows direct coefficient access (e.g. for a MatrixXd), this type is
       * either 'const Scalar&' or simply 'Scalar' for objects that do not allow direct coefficient access.
       */
      //using CoeffReturnType = typename Base::CoeffReturnType;
      //using CoeffReturnType = std::conditional_t<(Eigen::internal::traits<std::decay_t<NestedMatrix>>::Flags & Eigen::LvalueBit) != 0x0,
      //  const Scalar&, std::conditional_t<std::is_arithmetic_v<Scalar>, Scalar, const Scalar>>;


#ifdef __cpp_concepts
      template<typename Arg> requires (not std::same_as<std::decay_t<Arg>, EigenTensorWrapper>)
#else
      template<typename Arg, std::enable_if_t<(not std::is_same_v<std::decay_t<Arg>, EigenTensorWrapper>), int> = 0>
#endif
      explicit EigenTensorWrapper(Arg&& arg) : wrapped_expression {std::forward<Arg>(arg)} {}


      EIGEN_DEVICE_FUNC
      const auto dimensions() const
      {
        return make_dim(wrapped_expression, std::make_index_sequence<max_indices_of_v<NestedMatrix>>{});
      }


      /**
       * \brief Get the nested matrix.
       */
      decltype(auto) nested_matrix() & noexcept { return (wrapped_expression); }

      /// \overload
      decltype(auto) nested_matrix() const & noexcept { return (wrapped_expression); }

      /// \overload
      decltype(auto) nested_matrix() && noexcept { return (std::move(*this).wrapped_expression); }

      /// \overload
      decltype(auto) nested_matrix() const && noexcept { return (std::move(*this).wrapped_expression); }

    private:

      using ElementRef = decltype(OpenKalman::get_element(std::declval<NestedMatrix>(), 0, 0));
      static constexpr bool lvalue_get_element = std::is_same_v<ElementRef, std::decay_t<ElementRef>&>;

    public:

#ifdef __cpp_concepts
      auto& coeffRef(Index row, Index col) requires lvalue_get_element
#else
      template<bool b = lvalue_get_element, std::enable_if_t<b, int> = 0>
      auto& coeffRef(Index row, Index col)
#endif
      {
        return get_element(wrapped_expression, static_cast<std::size_t>(row), static_cast<std::size_t>(col));
      }


      constexpr decltype(auto) coeff(Index row, Index col) const
      {
        return get_element(wrapped_expression, static_cast<std::size_t>(row), static_cast<std::size_t>(col));
      }


#ifdef __cpp_concepts
      constexpr decltype(auto) data() requires directly_accessible<NestedMatrix>
#else
      template<typename T = NestedMatrix, std::enable_if_t<directly_accessible<T>, int> = 0>
      constexpr decltype(auto) data()
#endif
      {
        return internal::raw_data(wrapped_expression);
      }


#ifdef __cpp_concepts
      constexpr decltype(auto) data() const requires directly_accessible<NestedMatrix>
#else
      template<typename T = NestedMatrix, std::enable_if_t<directly_accessible<T>, int> = 0>
      constexpr decltype(auto) data() const
#endif
      {
        return internal::raw_data(wrapped_expression);
      }

    protected:

      NestedMatrix wrapped_expression;

    };


    /**
     * \brief Deduction guide for EigenWrapper
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires (not eigen_tensor_wrapper<Arg>)
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and not eigen_tensor_wrapper<Arg>, int> = 0>
#endif
    EigenTensorWrapper(Arg&&) -> EigenTensorWrapper<Arg>;


  } // namespace Eigen3


  // ------------ //
  //  Interfaces  //
  // ------------ //

  namespace interface
  {
    template<typename NestedMatrix, typename IndexType>
    struct IndexibleObjectTraits<Eigen3::EigenTensorWrapper<NestedMatrix, IndexType>>
    {
      static constexpr std::size_t max_indices = max_indices_of_v<NestedMatrix>;

      using index_type = index_type_of_t<NestedMatrix>;

      using scalar_type = scalar_type_of_t<NestedMatrix>;

      template<typename Arg, typename N>
      static constexpr auto get_index_descriptor(const Arg& arg, N n)
      {
        return OpenKalman::get_index_descriptor(nested_matrix(arg), n);
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one = one_by_one_matrix<NestedMatrix, b>;

      template<Likelihood b>
      static constexpr bool is_square = square_matrix<NestedMatrix, b>;

      static constexpr bool has_runtime_parameters = false;

      using type = std::tuple<NestedMatrix>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nested_matrix();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        return make_dense_writable_matrix_from(std::forward<Arg>(arg).nested_matrix());
      }

      template<typename Arg>
      static constexpr auto get_constant(const Arg& arg)
      {
        return constant_coefficient{arg.nested_matrix()};
      }

      template<typename Arg>
      static constexpr auto get_constant_diagonal(const Arg& arg)
      {
        return constant_diagonal_coefficient {arg.nested_matrix()};
      }

      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = triangular_matrix<NestedMatrix, t, b>;

      static constexpr bool is_triangular_adapter = false;

      static constexpr bool is_hermitian = hermitian_matrix<NestedMatrix, Likelihood::maybe>;

#ifdef __cpp_lib_concepts
      template<typename Arg, typename...I> requires element_gettable<decltype(nested_matrix(std::declval<Arg&&>())), sizeof...(I)>
#else
      template<typename Arg, typename...I, std::enable_if_t<element_gettable<decltype(nested_matrix(std::declval<Arg&&>())), sizeof...(I)>, int> = 0>
#endif
      static constexpr decltype(auto) get(Arg&& arg, I...i)
      {
        return get_element(nested_matrix(std::forward<Arg>(arg)), i...);
      }


#ifdef __cpp_lib_concepts
      template<typename Arg, typename Scalar, typename...I> requires element_settable<decltype(nested_matrix(std::declval<Arg&>())), sizeof...(I)>
#else
      template<typename Arg, typename Scalar, typename...I, std::enable_if_t<element_settable<decltype(nested_matrix(std::declval<Arg&>())), sizeof...(I)>, int> = 0>
#endif
      static constexpr void set(Arg& arg, const scalar_type_of_t<Arg>& s, I...i)
      {
        set_element(nested_matrix(arg), s, i...);
      }


      static constexpr bool is_writable =
        static_cast<bool>(Eigen::internal::traits<std::decay_t<NestedMatrix>>::Flags & (Eigen::LvalueBit | Eigen::DirectAccessBit));


#ifdef __cpp_lib_concepts
      template<typename Arg> requires directly_accessible<nested_matrix_of_t<Arg&>>
#else
      template<typename Arg, std::enable_if_t<directly_accessible<typename nested_matrix_of<Arg&>::type>, int> = 0>
#endif
      static constexpr auto*
      data(Arg& arg) { return internal::raw_data(arg.nested_matrix()); }


      static constexpr Layout layout = layout_of_v<NestedMatrix>;

    };

  } // namespace interface


} // namespace OpenKalman


namespace Eigen::internal
{
  // ------------------------- //
  //  Eigen::internal::traits  //
  // ------------------------- //

  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T, typename IndexType>
#else
    template<typename T, typename IndexType, typename = void>
#endif
    struct EigenTensorWrapperTraits
    {
    private:

      using ElementRef = decltype(OpenKalman::get_element(std::declval<T&>(), 0, 0));
      static constexpr bool lvalue_get_element = std::is_same_v<ElementRef, std::decay_t<ElementRef>&>;

    public:

      using Scalar = OpenKalman::scalar_type_of_t<T>;
      using StorageKind = Dense;
      using Index = IndexType;
      using Nested = T;
      static constexpr int NumDimensions = OpenKalman::max_indices_of_v<T>;
      static constexpr int Layout = ColMajor;
      template<typename U> struct MakePointer : Eigen::MakePointer<U> {};
      using PointerType = typename MakePointer<Scalar>::Type;
      enum {
        Flags = (lvalue_get_element ? LvalueBit : 0x0),
      };
    };


#ifdef __cpp_concepts
    template<OpenKalman::Eigen3::eigen_tensor_general T, typename IndexType> requires
      std::is_base_of_v<Eigen::TensorBase<std::decay_t<T>, ReadOnlyAccessors>, std::decay_t<T>>
    struct EigenTensorWrapperTraits<T, IndexType>
#else
    template<typename T, typename IndexType>
    struct EigenTensorWrapperTraits<T, IndexType, std::enable_if_t<OpenKalman::Eigen3::eigen_tensor_general<T> and
      std::is_base_of_v<Eigen::TensorBase<std::decay_t<T>, ReadOnlyAccessors>, std::decay_t<T>>>>
#endif
      : traits<T>
      {
        using Index = IndexType;
      };

  } // namespace detail


  template<typename T, typename IndexType>
  struct traits<OpenKalman::Eigen3::EigenTensorWrapper<T, IndexType>> : detail::EigenTensorWrapperTraits<std::decay_t<T>, IndexType> {};

} // namespace Eigen::internal


namespace Eigen
{
  // ------------------------ //
  //  Eigen::TensorEvaluator  //
  // ------------------------ //

  namespace detail
  {
#ifdef __cpp_concepts
    template<typename ArgType, typename IndexType, typename Device>
#else
    template<typename ArgType, typename IndexType, typename Device, typename = void>
#endif
    struct EigenTensorWrapperEvaluator : TensorEvaluator<std::decay_t<ArgType>, Device>
    {
      using Index = IndexType;
      using Scalar = OpenKalman::scalar_type_of_t<ArgType>;
      using CoeffReturnType = Scalar;
      using PacketReturnType = typename PacketType<CoeffReturnType, Device>::type;
      using XprType = OpenKalman::Eigen3::EigenTensorWrapper<ArgType, IndexType>;
      using Dimensions = typename XprType::Dimensions;
      static const int PacketSize =  PacketType<CoeffReturnType, Device>::size;
      using TensorPointerType = typename internal::traits<XprType>::template MakePointer<Scalar>::Type;
      using Storage = StorageMemory<Scalar, Device>;
      using EvaluatorPointerType = typename Storage::Type;

      // NumDimensions is -1 for variable dim tensors
      static const int NumCoords = OpenKalman::max_indices_of_v<ArgType> == OpenKalman::dynamic_size ? 0 : OpenKalman::max_indices_of_v<ArgType>;

      enum {
        IsAligned          = XprType::IsAligned,
        PacketAccess       = (PacketType<CoeffReturnType, Device>::size > 1),
        BlockAccess        = std::is_arithmetic_v<std::decay_t<Scalar>>,
        PreferBlockAccess  = false,
        Layout             = XprType::Layout,
        CoordAccess        = OpenKalman::max_indices_of_v<ArgType> > 0 and OpenKalman::max_indices_of_v<ArgType> != OpenKalman::dynamic_size,
        RawAccess          = static_cast<bool>(OpenKalman::directly_accessible<ArgType>),
      };
    };


#ifdef __cpp_concepts
    template<typename ArgType, typename IndexType, typename Device> requires OpenKalman::Eigen3::eigen_tensor_general<ArgType>
    struct EigenTensorWrapperEvaluator<ArgType, IndexType, Device>
#else
    template<typename ArgType, typename IndexType, typename Device>
    struct EigenTensorWrapperEvaluator<ArgType, IndexType, Device, std::enable_if_t<OpenKalman::Eigen3::eigen_tensor_general<ArgType>>>
#endif
      : TensorEvaluator<std::conditional_t<std::is_lvalue_reference_v<ArgType>, std::remove_reference_t<ArgType>,
        const std::remove_reference_t<ArgType>>, Device>
    {
      using XprType = std::conditional_t<std::is_lvalue_reference_v<ArgType>,
        std::remove_reference_t<ArgType>, const std::remove_reference_t<ArgType>>;
      using Base = TensorEvaluator<XprType, Device>;
      explicit EigenTensorWrapperEvaluator(const XprType& t, const Device& device) : Base {t, device} {}
    };

  } // namespace detail


  template<typename ArgType, typename IndexType, typename Device>
  struct TensorEvaluator<OpenKalman::Eigen3::EigenTensorWrapper<ArgType, IndexType>, Device>
    : detail::EigenTensorWrapperEvaluator<ArgType, IndexType, Device>
  {
    using XprType = OpenKalman::Eigen3::EigenTensorWrapper<ArgType, IndexType>;
    using Base = detail::EigenTensorWrapperEvaluator<ArgType, IndexType, Device>;
    explicit TensorEvaluator(const XprType& t, const Device& device) : Base {t.nested_matrix(), device} {}
  };

} // Eigen::internal

#endif //OPENKALMAN_EIGENTENSORWRAPPER_HPP